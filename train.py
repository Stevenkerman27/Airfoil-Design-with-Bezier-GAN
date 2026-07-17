import argparse
import csv
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import yaml
from torch.utils.data import DataLoader

from dataset import AirfoilDataset
from model import AerodynamicSurrogate, Discriminator, Generator
from plot_gan_metrics import METRIC_COLUMNS, plot_gan_metrics
from surrogate_split import load_split_indices, resolve_surrogate_dataset_config


SURROGATE_TARGET_ORDER = ['CM', 'CL', 'CD']
GAN_LABEL_ORDER = ['alpha', 'Re', 'CL', 'CM']
GAN_METRICS_PATH = 'model/gan_training_metrics.csv'
GAN_LOSS_PLOT_PATH = 'model/loss_curve.png'


def compute_gradient_penalty(discriminator, real_samples, fake_samples, conds, device):
    """Calculates the gradient penalty loss for WGAN-GP."""
    min_size = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_size]
    fake_samples = fake_samples[:min_size]
    conds = conds[:min_size]

    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates, conds)
    fake = torch.ones(real_samples.size(0), 1).to(device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty, grad_norm.mean().item()


def load_frozen_surrogate(config, device):
    _, dataset_config = resolve_surrogate_dataset_config(config)
    model = AerodynamicSurrogate(config).to(device)
    checkpoint = torch.load(
        dataset_config['best_model_path'],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_gan_auxiliary_stats(config, device):
    _, dataset_config = resolve_surrogate_dataset_config(config)
    gan_cond_norm = torch.load('model/cond_norm.pt', map_location=device, weights_only=True)
    gan_coord_norm = torch.load('model/coord_norm.pt', map_location=device, weights_only=True)
    surrogate_norm = torch.load(dataset_config['norm_path'], map_location=device, weights_only=True)

    surrogate_condition_names = surrogate_norm['condition']['names']
    surrogate_target_names = surrogate_norm['target']['names']
    if surrogate_condition_names != ['alpha', 'Re']:
        raise ValueError(f"Unexpected surrogate condition order: {surrogate_condition_names}")
    if surrogate_target_names != SURROGATE_TARGET_ORDER:
        raise ValueError(f"Unexpected surrogate target order: {surrogate_target_names}")

    return {
        'gan_cond_mean': gan_cond_norm['mean'].to(device),
        'gan_cond_std': gan_cond_norm['std'].to(device),
        'gan_coord': {
            'x_min': gan_coord_norm['x_min'].to(device),
            'x_max': gan_coord_norm['x_max'].to(device),
            'y_min': gan_coord_norm['y_min'].to(device),
            'y_max': gan_coord_norm['y_max'].to(device),
        },
        'surrogate_coord': {
            'x_min': surrogate_norm['coord']['x_min'].to(device),
            'x_max': surrogate_norm['coord']['x_max'].to(device),
            'y_min': surrogate_norm['coord']['y_min'].to(device),
            'y_max': surrogate_norm['coord']['y_max'].to(device),
        },
        'surrogate_condition_mean': surrogate_norm['condition']['mean'].to(device),
        'surrogate_condition_std': surrogate_norm['condition']['std'].to(device),
        'surrogate_target_mean': surrogate_norm['target']['mean'].to(device),
        'surrogate_target_std': surrogate_norm['target']['std'].to(device),
    }


def compute_generator_loss_weights(config, epoch):
    start_epoch = config['gan_aux_start_epoch']
    ramp_epochs = config['gan_aux_ramp_epochs']
    adv_final_weight = float(config['gan_adv_loss_final_weight'])
    surrogate_target_weight = float(config['gan_surrogate_loss_weight'])

    if start_epoch < 0:
        raise ValueError(f"gan_aux_start_epoch must be non-negative, got {start_epoch}")
    if ramp_epochs <= 0:
        raise ValueError(f"gan_aux_ramp_epochs must be positive, got {ramp_epochs}")
    if adv_final_weight < 0:
        raise ValueError(f"gan_adv_loss_final_weight must be non-negative, got {adv_final_weight}")
    if surrogate_target_weight < 0:
        raise ValueError(f"gan_surrogate_loss_weight must be non-negative, got {surrogate_target_weight}")

    if epoch < start_epoch:
        progress = 0.0
    elif epoch < start_epoch + ramp_epochs:
        progress = (epoch - start_epoch + 1) / ramp_epochs
    else:
        progress = 1.0

    progress = min(max(progress, 0.0), 1.0)
    return {
        'adv': 1.0 + progress * (adv_final_weight - 1.0),
        'surrogate': progress * surrogate_target_weight,
    }


def build_gan_surrogate_target_weights(config, device):
    weights = torch.tensor(
        config['gan_surrogate_target_loss_weights'],
        dtype=torch.float32,
        device=device,
    )
    if weights.numel() != 2:
        raise ValueError(
            'gan_surrogate_target_loss_weights must contain two values for [CM, CL]'
        )
    if torch.any(weights < 0):
        raise ValueError(
            'gan_surrogate_target_loss_weights must be non-negative, '
            f'got {weights.tolist()}'
        )
    if torch.sum(weights) <= 0:
        raise ValueError(
            'At least one gan_surrogate_target_loss_weights value must be positive, '
            f'got {weights.tolist()}'
        )
    return weights


def denormalize_gan_coords(coords, coord_stats, num_points):
    batch_size = coords.size(0)
    coords_2d = coords.view(batch_size, num_points, 2)
    x_values = coords_2d[:, :, 0] * (coord_stats['x_max'] - coord_stats['x_min'] + 1e-8) + coord_stats['x_min']
    y_values = coords_2d[:, :, 1] * (coord_stats['y_max'] - coord_stats['y_min'] + 1e-8) + coord_stats['y_min']
    return torch.stack([x_values, y_values], dim=2)


def normalize_surrogate_coords(physical_coords, coord_stats):
    x_values = (physical_coords[:, :, 0] - coord_stats['x_min']) / (coord_stats['x_max'] - coord_stats['x_min'] + 1e-8)
    y_values = (physical_coords[:, :, 1] - coord_stats['y_min']) / (coord_stats['y_max'] - coord_stats['y_min'] + 1e-8)
    return torch.stack([x_values, y_values], dim=2).view(physical_coords.size(0), -1)


def denormalize_gan_conditions(norm_conds, stats):
    return norm_conds * stats['gan_cond_std'] + stats['gan_cond_mean']


def build_surrogate_conditions(physical_conds, stats):
    condition = physical_conds[:, [0, 1]]
    return (condition - stats['surrogate_condition_mean']) / stats['surrogate_condition_std']


def compute_generator_auxiliary_losses(fake_foils, norm_conds, surrogate, stats, config):
    if surrogate is None or stats is None:
        raise ValueError('surrogate and stats must be loaded before computing auxiliary losses')

    num_points = config['num_output_points']
    physical_coords = denormalize_gan_coords(fake_foils, stats['gan_coord'], num_points)
    physical_conds = denormalize_gan_conditions(norm_conds, stats)

    surrogate_coords = normalize_surrogate_coords(physical_coords, stats['surrogate_coord'])
    surrogate_conditions = build_surrogate_conditions(physical_conds, stats)
    predicted_targets = surrogate(surrogate_coords, surrogate_conditions)

    target_cm_cl = torch.stack(
        [
            physical_conds[:, GAN_LABEL_ORDER.index('CM')],
            physical_conds[:, GAN_LABEL_ORDER.index('CL')],
        ],
        dim=1,
    )
    target_mean = stats['surrogate_target_mean'][[0, 1]]
    target_std = stats['surrogate_target_std'][[0, 1]]
    normalized_targets = (target_cm_cl - target_mean) / target_std
    per_target_losses = torch.mean(
        (predicted_targets[:, [0, 1]] - normalized_targets) ** 2,
        dim=0,
    )
    target_weights = build_gan_surrogate_target_weights(config, fake_foils.device)
    surrogate_loss = torch.mean(per_target_losses * target_weights)
    return surrogate_loss, per_target_losses[0], per_target_losses[1]


def save_checkpoint(generator, discriminator, epoch, path):
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def init_metrics_csv(path, append):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if not append or not file_exists:
            writer.writeheader()


def append_metrics_csv(path, metrics):
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writerow(metrics)


def run_lr_range_test(config, dataloader, device):
    print("--- Starting LR Range Test ---")

    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    lr_start = 1e-7
    lr_end = 1.0

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.0, 0.9))

    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2

    lambda_gp = config['lambda_gp']
    n_critic = config['n_critic']
    lr_mult = (lr_end / lr_start) ** (1 / total_steps)

    lrs = []
    d_losses_record = []
    g_losses_record = []

    beta = 0.2
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    initial_d_loss = None

    for i, (foils, conds) in enumerate(dataloader):
        foils = foils.to(device)
        conds = conds.to(device)
        batch_size = foils.size(0)

        current_lr = optimizer_D.param_groups[0]['lr']

        optimizer_D.zero_grad()
        z = torch.randn(batch_size, config['noise_dimension']).to(device)
        fake_foils = generator(z, conds)

        real_validity = discriminator(foils, conds)
        fake_validity = discriminator(fake_foils.detach(), conds)

        gradient_penalty, _ = compute_gradient_penalty(
            discriminator,
            foils,
            fake_foils.detach(),
            conds,
            device
        )

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        avg_d_loss = beta * avg_d_loss + (1 - beta) * d_loss.item()
        smoothed_d_loss = avg_d_loss / (1 - beta ** (i + 1))

        if i == 0:
            initial_d_loss = smoothed_d_loss

        if i > 0 and (abs(smoothed_d_loss) > abs(initial_d_loss) * 2 or math.isnan(smoothed_d_loss)):
            print(f"Loss diverged at step {i}, stopping LR test early.")
            break

        current_g_loss_val = 0.0
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z_gen = torch.randn(batch_size, config['noise_dimension']).to(device)
            fake_foil_gen = generator(z_gen, conds)
            fake_validity_gen = discriminator(fake_foil_gen, conds)
            g_loss = -torch.mean(fake_validity_gen)
            g_loss.backward()
            optimizer_G.step()
            current_g_loss_val = g_loss.item()
        elif len(g_losses_record) > 0:
            current_g_loss_val = g_losses_record[-1]

        avg_g_loss = beta * avg_g_loss + (1 - beta) * current_g_loss_val
        smoothed_g_loss = avg_g_loss / (1 - beta ** (i + 1))

        lrs.append(current_lr)
        d_losses_record.append(smoothed_d_loss)
        g_losses_record.append(smoothed_g_loss)

        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= lr_mult
        for param_group in optimizer_D.param_groups:
            param_group['lr'] *= lr_mult

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, d_losses_record, label='Smoothed D Loss')
    plt.plot(lrs, g_losses_record, label='Smoothed G Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plot_path = 'model/lr_range_test.png'
    plt.savefig(plot_path)
    plt.close()

    while True:
        try:
            user_lr = input(f"Please examine '{plot_path}' and enter the selected learning rate: ")
            final_lr = float(user_lr.strip())
            if final_lr > 0:
                break
        except ValueError:
            continue

    return final_lr


def train(resume_path=None):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device_cfg = config["device"]
    if device_cfg.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg.lower() == "cpu":
        device = torch.device("cpu")
    elif device_cfg.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown device configuration: {device_cfg}")
    print(f"Using device: {device}")

    batch_size = config['batch_size']
    dataset_name, dataset_config = resolve_surrogate_dataset_config(config)
    raw_data = torch.load(dataset_config['data_path'], weights_only=True)
    _, split_indices = load_split_indices(raw_data, config)
    dataset = AirfoilDataset(
        dataset_config['data_path'],
        split_indices['train'],
        'model/cond_norm.pt',
        'model/coord_norm.pt',
    )
    print(f"Using GAN training split '{dataset_name}' with {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = config['epochs']
    n_critic = config['n_critic']
    lambda_gp = config['lambda_gp']

    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    surrogate = None
    auxiliary_stats = None

    start_epoch = 0
    if resume_path:
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    lr = float(config['lr'])
    if lr <= 0.0:
        lr = run_lr_range_test(config, dataloader, device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=5e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=5e-5)

    init_metrics_csv(GAN_METRICS_PATH, append=resume_path is not None)

    import time
    for epoch in range(start_epoch, epochs):
        loss_weights = compute_generator_loss_weights(config, epoch)
        use_auxiliary_loss = loss_weights['surrogate'] > 0.0
        if use_auxiliary_loss and surrogate is None:
            surrogate = load_frozen_surrogate(config, device)
            auxiliary_stats = load_gan_auxiliary_stats(config, device)

        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_g_adv_loss = 0.0
        epoch_surrogate_loss = 0.0
        epoch_surrogate_cm_loss = 0.0
        epoch_surrogate_cl_loss = 0.0
        epoch_weighted_adv_loss = 0.0
        epoch_weighted_surrogate_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        epoch_grad_norm = 0.0
        batch_count = 0
        g_batch_count = 0

        for i, (foils, conds) in enumerate(dataloader):
            foils = foils.to(device)
            conds = conds.to(device)
            batch_size = foils.size(0)

            optimizer_D.zero_grad()

            z = torch.randn(batch_size, config['noise_dimension']).to(device)
            fake_foils = generator(z, conds)

            real_validity = discriminator(foils, conds)
            fake_validity = discriminator(fake_foils.detach(), conds)

            gradient_penalty, grad_norm = compute_gradient_penalty(
                discriminator,
                foils,
                fake_foils.detach(),
                conds,
                device
            )

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            epoch_d_loss += d_loss.item()
            epoch_real_score += torch.mean(real_validity).item()
            epoch_fake_score += torch.mean(fake_validity).item()
            epoch_grad_norm += grad_norm
            batch_count += 1

            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z_gen = torch.randn(batch_size, config['noise_dimension']).to(device)
                fake_foil = generator(z_gen, conds)

                fake_validity_gen = discriminator(fake_foil, conds)
                g_adv_loss = -torch.mean(fake_validity_gen)
                if use_auxiliary_loss:
                    surrogate_loss, surrogate_cm_loss, surrogate_cl_loss = compute_generator_auxiliary_losses(
                        fake_foil,
                        conds,
                        surrogate,
                        auxiliary_stats,
                        config,
                    )
                else:
                    surrogate_loss = torch.zeros((), device=device)
                    surrogate_cm_loss = torch.zeros((), device=device)
                    surrogate_cl_loss = torch.zeros((), device=device)

                g_loss = (
                    loss_weights['adv'] * g_adv_loss
                    + loss_weights['surrogate'] * surrogate_loss
                )
                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                epoch_g_adv_loss += g_adv_loss.item()
                epoch_surrogate_loss += surrogate_loss.item()
                epoch_surrogate_cm_loss += surrogate_cm_loss.item()
                epoch_surrogate_cl_loss += surrogate_cl_loss.item()
                epoch_weighted_adv_loss += (loss_weights['adv'] * g_adv_loss).item()
                epoch_weighted_surrogate_loss += (loss_weights['surrogate'] * surrogate_loss).item()
                g_batch_count += 1

        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_g_adv_loss = epoch_g_adv_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_surrogate_loss = epoch_surrogate_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_surrogate_cm_loss = epoch_surrogate_cm_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_surrogate_cl_loss = epoch_surrogate_cl_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_weighted_adv_loss = epoch_weighted_adv_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_weighted_surrogate_loss = epoch_weighted_surrogate_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_real_score = epoch_real_score / batch_count
        avg_fake_score = epoch_fake_score / batch_count
        avg_grad_norm = epoch_grad_norm / batch_count

        epoch_duration = time.time() - start_time
        append_metrics_csv(
            GAN_METRICS_PATH,
            {
                'epoch': epoch,
                'd_loss': avg_d_loss,
                'g_loss_total': avg_g_loss,
                'g_adv_raw': avg_g_adv_loss,
                'surrogate_cm_raw': avg_surrogate_cm_loss,
                'surrogate_cl_raw': avg_surrogate_cl_loss,
                'surrogate_raw': avg_surrogate_loss,
                'g_adv_weighted': avg_weighted_adv_loss,
                'surrogate_weighted': avg_weighted_surrogate_loss,
                'real_score': avg_real_score,
                'fake_score': avg_fake_score,
                'grad_norm': avg_grad_norm,
                'w_adv': loss_weights['adv'],
                'w_surrogate': loss_weights['surrogate'],
            },
        )

        if epoch % 2 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Time: {epoch_duration:.2f}s] "
                f"[D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}] "
                f"[G adv: {avg_g_adv_loss:.4f}] [Surr: {avg_surrogate_loss:.4f}] "
                f"[CM: {avg_surrogate_cm_loss:.4f}] [CL: {avg_surrogate_cl_loss:.4f}] "
                f"[W adv: {loss_weights['adv']:.3f}] "
                f"[W surr: {loss_weights['surrogate']:.3f}]"
            )

        if epoch % 5 == 0 and epoch > 0:
            print(
                f"[Critic Real: {avg_real_score:.4f}] "
                f"[Critic Fake: {avg_fake_score:.4f}] [GP Norm: {avg_grad_norm:.4f}]"
            )

    save_checkpoint(generator, discriminator, epochs - 1, "model/gan_final.pt")
    plot_gan_metrics(GAN_METRICS_PATH, GAN_LOSS_PLOT_PATH)
    print("Training finished and final model saved to model/gan_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CWGAN-GP for airfoil design")
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint (.pt)")
    args = parser.parse_args()
    train(resume_path=args.resume)
