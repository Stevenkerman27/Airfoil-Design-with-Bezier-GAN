import yaml
import sys
import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import argparse
from model import Generator, Discriminator
from dataset import AirfoilDataset
from foildata.xfoil import run_xfoil_single
from utils import calculate_relative_thickness, check_intersection, check_shape_intersections

def compute_gradient_penalty(D, real_samples, fake_samples, conds, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    min_size = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_size]
    fake_samples = fake_samples[:min_size]
    conds = conds[:min_size]

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates, _ = D(interpolates, conds)
    
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    # Get gradient w.r.t. interpolates
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

import concurrent.futures

def _evaluate_single(args):
    """Worker function to evaluate a single airfoil and return its physical error."""
    i, coords, alpha, reynolds, target_cl, target_t = args

    # Quick geometric bounds check
    x = coords[:, 0]
    y = coords[:, 1]

    # 1. x bounds (must be roughly in [0, 1], with some tolerance for Bezier curve overshoots)
    if np.any(x < -0.1) or np.any(x > 1.2):
        return i, 2.0  # High penalty

    # 2. y bounds (airfoils thicker than ~40% (y = +/-0.2) are extremely rare and likely unrealistic)
    if np.any(np.abs(y) > 0.2):
        return i, 2.0

    # Check for self-intersection and shape constraints
    if check_intersection(coords) or check_shape_intersections(coords):
        return i, 2.0

    # Calculate thickness
    calc_t = calculate_relative_thickness(coords)
    t_res = abs(calc_t - target_t) / (abs(target_t) + 1e-8)

    # Calculate Cl via Xfoil
    xfoil_res = run_xfoil_single(coords, reynolds, alpha, return_all=True)
    if xfoil_res is None:
        return i, 2.0

    calc_cl = xfoil_res.get('CL', np.nan)
    calc_cd = xfoil_res.get('CD', np.nan)
    calc_cm = xfoil_res.get('CM', np.nan)

    if np.isnan(calc_cl) or np.isnan(calc_cd) or np.isnan(calc_cm):
        return i, 2.0

    cl_res = abs(calc_cl - target_cl) / (abs(target_cl) + 1e-8)

    total_err = cl_res + t_res
    return i, min(total_err, 2.0)
def evaluate_physics(fake_foils, conds, norm_stats, coord_norm_stats, max_workers=16):
    """
    Evaluates generated foils using XFOIL and returns continuous physical errors.
    """
    batch_size = fake_foils.size(0)
    num_pts = fake_foils.size(1) // 2
    
    # Un-normalize conditions: [alpha, Re, CL, Thickness]
    y_mean = norm_stats['mean'].to(conds.device)
    y_std = norm_stats['std'].to(conds.device)
    real_conds = conds * y_std + y_mean
    
    # Un-normalize airfoil coordinates
    x_min = coord_norm_stats['x_min'].to(fake_foils.device)
    x_max = coord_norm_stats['x_max'].to(fake_foils.device)
    y_min = coord_norm_stats['y_min'].to(fake_foils.device)
    y_max = coord_norm_stats['y_max'].to(fake_foils.device)
    
    # Batch un-normalization to avoid CPU-GPU sync per item
    coords_unnorm = fake_foils.detach().clone().view(batch_size, -1, 2)
    coords_unnorm[:, :, 0] = coords_unnorm[:, :, 0] * (x_max - x_min + 1e-8) + x_min
    coords_unnorm[:, :, 1] = coords_unnorm[:, :, 1] * (y_max - y_min + 1e-8) + y_min
    
    # Prepare arguments for the worker pool
    eval_args = []
    
    coords_np_all = coords_unnorm.cpu().numpy()
    
    real_conds_np = real_conds.cpu().numpy()
    
    for i in range(batch_size):
        coords_np = coords_np_all[i]
        alpha = real_conds_np[i, 0].item()
        reynolds = real_conds_np[i, 1].item()
        target_cl = real_conds_np[i, 2].item()
        target_t = real_conds_np[i, 3].item()
        
        eval_args.append((i, coords_np, alpha, reynolds, target_cl, target_t))

    # Run evaluations concurrently
    real_errs = [2.0] * batch_size # Default to high penalty
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_evaluate_single, eval_args))
        
    for idx, err in results:
        real_errs[idx] = err
            
    return torch.tensor(real_errs, dtype=torch.float32, device=fake_foils.device).view(-1, 1)

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

    # Loss Plot
    ax1.plot(epochs_x, d_losses, label="D Loss")
    ax1.plot(epochs_x, g_losses, label="G Loss")
    ax1.set_title("WGAN-GP Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Critic Scores Plot
    ax2.plot(epochs_x, real_scores, label="Critic Real Score")
    ax2.plot(epochs_x, fake_scores, label="Critic Fake Score")
    ax2.set_title("Discriminator Scores")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    # Gradient Penalty Norm Plot
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

import math

def run_lr_range_test(config, dataloader, device, generator=None, discriminator=None, phase=1):
    print(f"--- Starting LR Range Test (Phase {phase}) ---")
    
    if generator is None or discriminator is None:
        generator = Generator(config).to(device)
        discriminator = Discriminator(config).to(device)
        restore_state = False
    else:
        gen_state = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        dis_state = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
        restore_state = True
    
    lr_start = 1e-7
    lr_end = 1.0  # WGAN 通常测到 1.0 足够发现崩溃点
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    
    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2
        
    lambda_gp = config['lambda_gp']
    n_critic = config['n_critic']
    
    # 采用指数级（按倍数）增长
    lr_mult = (lr_end / lr_start) ** (1 / total_steps)
    
    lrs = []
    d_losses_record = []
    g_losses_record = []
    
    # 引入 EMA 平滑变量
    beta = 0.2
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    best_d_loss = float('inf')
    initial_d_loss = None
    
    for i, (foils, conds) in enumerate(dataloader):
        foils = foils.to(device)
        conds = conds.to(device)
        batch_size = foils.size(0)
        
        # 获取当前学习率用于记录
        current_lr = optimizer_D.param_groups[0]['lr']
        
        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, config['noise_dimension']).to(device)
        fake_foils = generator(z, conds)
        
        real_validity, _ = discriminator(foils, conds)
        fake_validity, _ = discriminator(fake_foils.detach(), conds)
        
        gradient_penalty, _ = compute_gradient_penalty(
            discriminator, foils, fake_foils.detach(), conds, device
        )
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        
        # 计算 EMA 平滑 D Loss
        avg_d_loss = beta * avg_d_loss + (1 - beta) * d_loss.item()
        smoothed_d_loss = avg_d_loss / (1 - beta ** (i + 1))

        if i == 0:
            initial_d_loss = smoothed_d_loss
        
        # 防爆机制。如果发散，立即停止，保护图表比例尺
        if i > 0 and (abs(smoothed_d_loss) > abs(initial_d_loss) * 2 or math.isnan(smoothed_d_loss)):
            print(f"Loss diverged at step {i}, stopping LR test early.")
            break
            
        if smoothed_d_loss < best_d_loss:
            best_d_loss = smoothed_d_loss
            
        # --- Train Generator ---
        current_g_loss_val = 0.0
        # 遵循 n_critic 设定
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z_gen = torch.randn(batch_size, config['noise_dimension']).to(device)
            fake_foil_gen = generator(z_gen, conds)
            fake_validity_gen, _ = discriminator(fake_foil_gen, conds)
            g_loss = -torch.mean(fake_validity_gen)
            g_loss.backward()
            optimizer_G.step()
            current_g_loss_val = g_loss.item()
        else:
            current_g_loss_val = g_losses_record[-1] if len(g_losses_record) > 0 else 0.0

        # 平滑 G loss
        avg_g_loss = beta * avg_g_loss + (1 - beta) * current_g_loss_val
        smoothed_g_loss = avg_g_loss / (1 - beta ** (i + 1))
        
        lrs.append(current_lr)
        d_losses_record.append(smoothed_d_loss)
        g_losses_record.append(smoothed_g_loss)
        
        # 更新学习率 (乘以常数)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= lr_mult
        for param_group in optimizer_D.param_groups:
            param_group['lr'] *= lr_mult
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, d_losses_record, label='Smoothed D Loss')
    plt.plot(lrs, g_losses_record, label='Smoothed G Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title(f'LR Range Test (Phase {phase})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plot_path = f'model/lr_range_test_phase{phase}.png'
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
            
    if restore_state:
        generator.load_state_dict(gen_state)
        discriminator.load_state_dict(dis_state)
        
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
    
    # Dataset & DataLoader
    batch_size = config['batch_size']
    dataset = AirfoilDataset("model/airfoil_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = config['epochs']
    pre_train_epoch = config['pre_train_epoch']
    n_critic = config['n_critic']
    lambda_gp = config['lambda_gp']
    
    # Load norm stats
    norm_stats = torch.load("model/cond_norm.pt", map_location=device, weights_only=True)
    coord_norm_stats = torch.load("model/coord_norm.pt", map_location=device, weights_only=True)
    max_workers = config['max_workers']

    # Initialize formal models
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    start_epoch = 0
    if resume_path:
        print(f"Loading pre-trained model from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        # Jump to physics training phase
        start_epoch = pre_train_epoch
        print(f"Skipping pre-training, starting from epoch {start_epoch}")

    # --- Phase 1 LR ---
    phase1_lr = float(config['lr'])
    if phase1_lr <= 0.0 and start_epoch < pre_train_epoch:
        phase1_lr = run_lr_range_test(config, dataloader, device, phase=1)
    elif start_epoch >= pre_train_epoch:
        phase1_lr = 1e-4  # Dummy value, will be overridden immediately

    # Optimizers for formal training
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=phase1_lr, betas=(0.0, 0.9), weight_decay=5e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=phase1_lr, betas=(0.0, 0.9), weight_decay=5e-5)
    
    eps_cl_start = config['eps_cl_start']
    eps_cl_end = config['eps_cl_end']
    eps_t_start = config['eps_t_start']
    eps_t_end = config['eps_t_end']

    # Lists to keep track of progress
    d_losses = []
    g_losses = []
    real_scores = []
    fake_scores = []
    grad_norms = []
    validity_history = []

    import time
    for epoch in range(start_epoch, epochs):
        if epoch == pre_train_epoch:
            phase2_lr = float(config['lr_phase2'])
            if phase2_lr <= 0.0:
                print("\n--- Transitioning to Phase 2, determining learning rate ---")
                phase2_lr = run_lr_range_test(config, dataloader, device, generator=generator, discriminator=discriminator, phase=2)
            
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = phase2_lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = phase2_lr
            print(f"Phase 2 learning rate set to {phase2_lr}")

        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        epoch_grad_norm = 0.0
        batch_count = 0
        g_batch_count = 0
        
        total_samples = 0
        total_valid = 0
        
        # Calculate current epsilons for physics evaluation (only used for logging validity rate now)
        curr_eps_cl = eps_cl_start
        curr_eps_t = eps_t_start
        if epoch >= pre_train_epoch:
            if epochs > pre_train_epoch + 1:
                progress = (epoch - pre_train_epoch) / (epochs - pre_train_epoch - 1)
                curr_eps_cl = eps_cl_start - (eps_cl_start - eps_cl_end) * progress
                curr_eps_t = eps_t_start - (eps_t_start - eps_t_end) * progress
            else:
                curr_eps_cl = eps_cl_end
                curr_eps_t = eps_t_end

        for i, (foils, conds) in enumerate(dataloader):
            foils = foils.to(device)
            conds = conds.to(device)
            batch_size = foils.size(0)
            total_samples += batch_size

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate a batch of fake foils
            z = torch.randn(batch_size, config['noise_dimension']).to(device)
            fake_foils = generator(z, conds)

            # Evaluate Physics to get continuous error for fake foils
            if epoch >= pre_train_epoch:
                real_errs = evaluate_physics(fake_foils, conds, norm_stats, coord_norm_stats, max_workers=max_workers)
                # Keep tracking validity just for logs
                total_valid += torch.sum(real_errs < (curr_eps_cl + curr_eps_t)).item()
            else:
                real_errs = torch.zeros((batch_size, 1)).to(device)

            real_validity, _ = discriminator(foils, conds)
            fake_validity, fake_pred_err = discriminator(fake_foils.detach(), conds)

            # Gradient penalty (Always standard GP on full batch now)
            gradient_penalty, grad_norm = compute_gradient_penalty(
                discriminator,
                foils,
                fake_foils.detach(),
                conds,
                device
            )

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty      

            # Physics continuous prediction loss
            if epoch >= pre_train_epoch:
                lambda_phys_d = config['lambda_phys_d']
                mse_loss = torch.nn.MSELoss()(fake_pred_err, real_errs)
                d_loss += lambda_phys_d * mse_loss

            d_loss.backward()
            optimizer_D.step()

            epoch_d_loss += d_loss.item()
            epoch_real_score += torch.mean(real_validity).item()
            epoch_fake_score += torch.mean(fake_validity).item()
            epoch_grad_norm += grad_norm
            batch_count += 1

            # -----------------
            #  Train Generator
            # -----------------
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z_gen = torch.randn(batch_size, config['noise_dimension']).to(device)
                fake_foil = generator(z_gen, conds)

                fake_validity_gen, fake_pred_err_gen = discriminator(fake_foil, conds)
                g_loss = -torch.mean(fake_validity_gen)

                if epoch >= pre_train_epoch:
                    lambda_phys_g = config['lambda_phys_g']
                    g_loss += lambda_phys_g * torch.mean(fake_pred_err_gen)

                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                g_batch_count += 1                
        # Calculate average loss and validity for the epoch
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / g_batch_count if g_batch_count > 0 else 0
        avg_real_score = epoch_real_score / batch_count
        avg_fake_score = epoch_fake_score / batch_count
        avg_grad_norm = epoch_grad_norm / batch_count
        avg_validity = total_valid / total_samples
        
        epoch_duration = time.time() - start_time
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        grad_norms.append(avg_grad_norm)
        validity_history.append(avg_validity)

        if epoch % 2 == 0:
            speed_msg = f"[Epoch {epoch}/{epochs}] [Time: {epoch_duration:.2f}s] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
            if epoch >= pre_train_epoch:
                speed_msg += f" [Validity: {avg_validity:.2%}]"
            print(speed_msg)

        if epoch % 5 == 0 and epoch > 0:
            print(f"[Critic Real: {avg_real_score:.4f}] [Critic Fake: {avg_fake_score:.4f}] [GP Norm: {avg_grad_norm:.4f}]")

        # Check if pre-training is finished and save checkpoint
        if epoch == pre_train_epoch - 1:
            save_checkpoint(generator, discriminator, epoch, "model/pre_train.pt")
            plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, "model/pre_train_loss.png")
            print(f"Pre-training finished. Checkpoint and plot saved.")

    # Save models
    save_checkpoint(generator, discriminator, epochs - 1, "model/gan_final.pt")
    plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, "model/loss_curve.png")
    print("Training finished and final model saved to model/gan_final.pt")

    # Plot and save final results
    # (Removed old plotting code as plot_metrics is now used)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CWGAN-GP for airfoil design")
    parser.add_argument("--resume", "-r", type=str, help="Path to pre-trained model checkpoint (.pt)")
    args = parser.parse_args()
    train(resume_path=args.resume)
