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
    
    d_interpolates = D(interpolates, conds)
    
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
    """Worker function to evaluate a single airfoil."""
    i, coords, alpha, reynolds, target_cl, target_t, eps = args

    # Quick geometric bounds check
    x = coords[:, 0]
    y = coords[:, 1]
    
    # 1. x bounds (must be roughly in [0, 1], with some tolerance for Bezier curve overshoots)
    if np.any(x < -0.1) or np.any(x > 1.2):
        return i, False
        
    # 2. y bounds (airfoils thicker than ~40% (y = +/-0.2) are extremely rare and likely unrealistic)
    if np.any(np.abs(y) > 0.2):
        return i, False

    # Check for self-intersection and shape constraints
    if check_intersection(coords) or check_shape_intersections(coords):
        return i, False
    
    # Calculate thickness
    calc_t = calculate_relative_thickness(coords)
    t_res = abs(calc_t - target_t) / (abs(target_t) + 1e-8)
    
    if t_res > eps:
        return i, False
        
    # Calculate Cl via Xfoil
    calc_cl = run_xfoil_single(coords, reynolds, alpha)
    if calc_cl is None:
        return i, False
        
    cl_res = abs(calc_cl - target_cl) / (abs(target_cl) + 1e-8)
    
    if cl_res <= eps:
        return i, True
    else:
        return i, False

def evaluate_physics(fake_foils, conds, norm_stats, eps):
    """
    Evaluates generated foils using physics models.
    Splits indices into R_eps (reasonable) and F_eps (unreasonable).
    Uses ThreadPoolExecutor to evaluate multiple foils concurrently.
    """
    batch_size = fake_foils.size(0)
    num_pts = fake_foils.size(1) // 2
    r_idx = []
    f_idx = []
    
    # Un-normalize conditions: [alpha, Re, CL, Thickness]
    y_mean = norm_stats['mean'].to(conds.device)
    y_std = norm_stats['std'].to(conds.device)
    real_conds = conds * y_std + y_mean
    
    # Prepare arguments for the worker pool
    eval_args = []
    for i in range(batch_size):
        coords_flat = fake_foils[i].detach().cpu().numpy()
        coords = coords_flat.reshape(num_pts, 2)
        alpha = real_conds[i, 0].item()
        reynolds = real_conds[i, 1].item()
        target_cl = real_conds[i, 2].item()
        target_t = real_conds[i, 3].item()
        
        eval_args.append((i, coords, alpha, reynolds, target_cl, target_t, eps))

    # Run evaluations concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        results = list(executor.map(_evaluate_single, eval_args))
        
    for idx, is_reasonable in results:
        if is_reasonable:
            r_idx.append(idx)
        else:
            f_idx.append(idx)
            
    return r_idx, f_idx

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

def run_lr_range_test(config, dataloader, device):
    print("--- Starting LR Range Test ---")
    
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    
    lr_start = 1e-7
    lr_end = 1.0  # WGAN 通常测到 1.0 足够发现崩溃点
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    
    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2
        
    lambda_gp = config.get('lambda_gp')
    n_critic = config.get('n_critic')
    
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
        z = torch.randn(batch_size, config.get('noise_dimension')).to(device)
        fake_foils = generator(z, conds)
        
        real_validity = discriminator(foils, conds)
        fake_validity = discriminator(fake_foils.detach(), conds)
        
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
            z_gen = torch.randn(batch_size, config.get('noise_dimension')).to(device)
            fake_foil_gen = generator(z_gen, conds)
            fake_validity_gen = discriminator(fake_foil_gen, conds)
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
    plt.title('LR Range Test')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('model/lr_range_test.png')
    plt.close()
    
    while True:
        try:
            user_lr = input("Please examine 'model/lr_range_test.png' and enter the selected learning rate: ")
            final_lr = float(user_lr.strip())
            if final_lr > 0:
                break
        except ValueError:
            pass
            
    return final_lr

def train(resume_path=None):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device_cfg = config.get("device", "auto")
    if device_cfg.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset & DataLoader
    batch_size = config.get('batch_size', 16)
    dataset = AirfoilDataset("model/airfoil_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = config.get('epochs')
    pre_train_epoch = config.get('pre_train_epoch')
    n_critic = config.get('n_critic')
    lambda_gp = config.get('lambda_gp')
    
    # Load norm stats
    norm_stats = torch.load("model/cond_norm.pt", map_location=device, weights_only=True)

    # --- Run LR Range Test ---
    selected_lr = run_lr_range_test(config, dataloader, device)
    
    # Initialize formal models
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    start_epoch = 0
    if resume_path:
        print(f"Loading pre-trained model from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        # Jump to physics training phase
        start_epoch = pre_train_epoch
        print(f"Skipping pre-training, starting from epoch {start_epoch}")

    # Optimizers for formal training
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=selected_lr, betas=(0.0, 0.9), weight_decay=5e-5)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=selected_lr, betas=(0.0, 0.9), weight_decay=5e-5)
    eps_start = config.get('eps_start')
    eps_end = config.get('eps_end')

    # Lists to keep track of progress
    d_losses = []
    g_losses = []
    real_scores = []
    fake_scores = []
    grad_norms = []
    validity_history = []

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
        
        total_samples = 0
        total_valid = 0
        
        # Calculate current epsilon for physics evaluation (moved outside batch loop)
        current_eps = eps_start
        if epoch >= pre_train_epoch:
            if epochs > pre_train_epoch + 1:
                progress = (epoch - pre_train_epoch) / (epochs - pre_train_epoch - 1)
                current_eps = eps_start - (eps_start - eps_end) * progress
            else:
                current_eps = eps_end

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
            z = torch.randn(batch_size, config.get('noise_dimension')).to(device)
            fake_foils = generator(z, conds)

            # Evaluate Physics to get R_eps and F_eps indices
            if epoch < pre_train_epoch:
                r_idx = []
                f_idx = list(range(batch_size))
            else:
                r_idx, f_idx = evaluate_physics(fake_foils, conds, norm_stats, current_eps)
                total_valid += len(r_idx)

            # Split fake_foils
            if len(r_idx) > 0:
                r_foils = fake_foils[r_idx].detach()
                r_conds = conds[r_idx]
                combined_real_foils = torch.cat([foils, r_foils], dim=0)
                combined_real_conds = torch.cat([conds, r_conds], dim=0)
            else:
                combined_real_foils = foils
                combined_real_conds = conds

            if len(f_idx) > 0:
                f_foils = fake_foils[f_idx].detach()
                f_conds = conds[f_idx]
            else:
                # If no unreasonable foils, it means the generator is performing perfectly for this batch
                print("\n[Early Stopping] No unreasonable airfoils found in discriminator step. Training complete.")
                save_checkpoint(generator, discriminator, epoch, "model/gan_final.pt")
                plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, "model/loss_curve.png")
                sys.exit(0)

            real_validity = discriminator(combined_real_foils, combined_real_conds)
            fake_validity = discriminator(f_foils, f_conds)
            
            # Gradient penalty
            gradient_penalty, grad_norm = compute_gradient_penalty(
                discriminator, 
                combined_real_foils, 
                f_foils[:combined_real_foils.size(0)],
                combined_real_conds[:combined_real_foils.size(0)], 
                device
            )

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

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

                z_gen = torch.randn(batch_size, config.get('noise_dimension')).to(device)
                fake_foil = generator(z_gen, conds)
                
                # Evaluate to find F_eps for generator optimization
                if epoch < pre_train_epoch:
                    f_idx_gen = list(range(batch_size))
                else:
                    _, f_idx_gen = evaluate_physics(fake_foil, conds, norm_stats, current_eps)
                
                if len(f_idx_gen) > 0:
                    f_foil_gen = fake_foil[f_idx_gen]
                    f_conds_gen = conds[f_idx_gen]
                    
                    fake_validity_gen = discriminator(f_foil_gen, f_conds_gen)
                    g_loss = -torch.mean(fake_validity_gen)

                    g_loss.backward()
                    optimizer_G.step()
                    
                    epoch_g_loss += g_loss.item()
                    g_batch_count += 1
                else:
                    # All samples physically reasonable, terminate training as the goal is reached
                    print("\n[Early Stopping] All generated samples are physically reasonable. Training complete.")
                    # Save final models before exiting
                    save_checkpoint(generator, discriminator, epoch, "model/gan_final.pt")
                    plot_metrics(d_losses, g_losses, real_scores, fake_scores, grad_norms, "model/loss_curve.png")
                    import sys
                    sys.exit(0)
                
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
