import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import yaml
from torch.utils.data import DataLoader

from dataset import AirfoilDataset
from model import Discriminator, Generator


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


def save_checkpoint(generator, discriminator, epoch, path):
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, path):
    epochs_x = np.arange(len(d_losses))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)

    ax1.plot(epochs_x, d_losses, label="D Loss")
    ax1.plot(epochs_x, g_losses, label="G Loss")
    ax1.set_title("WGAN-GP Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_x, real_scores, label="Critic Real Score")
    ax2.plot(epochs_x, fake_scores, label="Critic Fake Score")
    ax2.set_title("Discriminator Scores")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs_x, grad_norms, label="GP Norm", color='orange')
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax3.set_title("Gradient Penalty Norm")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Norm")
    ax3.legend()
    ax3.grid(True)

    plt.savefig(path)
    plt.close()
    print(f"Training plots saved to {path}")


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
    dataset = AirfoilDataset("model/airfoil_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = config['epochs']
    n_critic = config['n_critic']
    lambda_gp = config['lambda_gp']

    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

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

    d_losses = []
    g_losses = []
    real_scores = []
    fake_scores = []
    grad_norms = []

    import time
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
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
                g_loss = -torch.mean(fake_validity_gen)
                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                g_batch_count += 1

        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / g_batch_count if g_batch_count > 0 else 0.0
        avg_real_score = epoch_real_score / batch_count
        avg_fake_score = epoch_fake_score / batch_count
        avg_grad_norm = epoch_grad_norm / batch_count

        epoch_duration = time.time() - start_time
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        grad_norms.append(avg_grad_norm)

        if epoch % 2 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Time: {epoch_duration:.2f}s] "
                f"[D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
            )

        if epoch % 5 == 0 and epoch > 0:
            print(
                f"[Critic Real: {avg_real_score:.4f}] "
                f"[Critic Fake: {avg_fake_score:.4f}] [GP Norm: {avg_grad_norm:.4f}]"
            )

    save_checkpoint(generator, discriminator, epochs - 1, "model/gan_final.pt")
    plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, "model/loss_curve.png")
    print("Training finished and final model saved to model/gan_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CWGAN-GP for airfoil design")
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint (.pt)")
    args = parser.parse_args()
    train(resume_path=args.resume)
