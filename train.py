import yaml
import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from dataset import AirfoilDataset
from foildata.xfoil import run_xfoil_single
from utils import calculate_relative_thickness, check_intersection


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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def evaluate_physics(fake_foils, conds, norm_stats, eps):
    """
    Evaluates generated foils using physics models.
    Splits indices into R_eps (reasonable) and F_eps (unreasonable).
    """
    batch_size = fake_foils.size(0)
    num_pts = fake_foils.size(1) // 2
    r_idx = []
    f_idx = []
    
    # Un-normalize conditions: [alpha, Re, CL, Thickness]
    y_mean = norm_stats['mean'].to(conds.device)
    y_std = norm_stats['std'].to(conds.device)
    real_conds = conds * y_std + y_mean
    
    for i in range(batch_size):
        coords_flat = fake_foils[i].detach().cpu().numpy()
        coords = coords_flat.reshape(num_pts, 2)
        
        alpha = real_conds[i, 0].item()
        reynolds = real_conds[i, 1].item()
        target_cl = real_conds[i, 2].item()
        target_t = real_conds[i, 3].item()

        # Quick geometric bounds check
        x = coords[:, 0]
        y = coords[:, 1]
        
        # 1. x bounds (must be roughly in [0, 1], with some tolerance for Bezier curve overshoots)
        if np.any(x < -0.1) or np.any(x > 1.2):
            f_idx.append(i)
            continue
            
        # 2. y bounds (airfoils thicker than ~40% (y = +/-0.2) are extremely rare and likely unrealistic)
        if np.any(np.abs(y) > 0.2):
            f_idx.append(i)
            continue

        # Check for self-intersection
        if check_intersection(coords):
            f_idx.append(i)
            continue
        
        # Calculate thickness
        calc_t = calculate_relative_thickness(coords)
        t_res = abs(calc_t - target_t) / (abs(target_t) + 1e-8)
        
        if t_res > eps:
            f_idx.append(i)
            continue
            
        # Calculate Cl via Xfoil
        calc_cl = run_xfoil_single(coords, reynolds, alpha)
        if calc_cl is None:
            f_idx.append(i)
            continue
            
        cl_res = abs(calc_cl - target_cl) / (abs(target_cl) + 1e-8)
        
        if cl_res <= eps:
            r_idx.append(i)
        else:
            f_idx.append(i)
            
    return r_idx, f_idx

def train():
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

    # Initialize models
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    # Optimizers (WGAN-GP uses Adam with beta1=0, beta2=0.9 usually)
    lr = float(config.get('lr', 1e-4))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    epochs = config.get('epochs', 120)
    n_critic = config.get('n_critic', 3)
    lambda_gp = config.get('lambda_gp', 10)
    
    # Load norm stats
    norm_stats = torch.load("model/cond_norm.pt", map_location=device, weights_only=True)
    eps_start = config.get('eps_start', 0.10)
    eps_end = config.get('eps_end', 0.02)

    # Lists to keep track of progress
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        g_batch_count = 0
        
        for i, (foils, conds) in enumerate(dataloader):
            foils = foils.to(device)
            conds = conds.to(device)
            batch_size = foils.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Calculate current epsilon
            current_eps = eps_start - (eps_start - eps_end) * (epoch / epochs)

            # Generate a batch of fake foils
            z = torch.randn(batch_size, config.get('noise_dimension')).to(device)
            fake_foils = generator(z, conds)

            # Evaluate Physics to get R_eps and F_eps indices
            r_idx, f_idx = evaluate_physics(fake_foils, conds, norm_stats, current_eps)

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
                # If no unreasonable foils, fake_foils acts as F_eps to keep training going
                f_foils = fake_foils.detach()
                f_conds = conds

            real_validity = discriminator(combined_real_foils, combined_real_conds)
            fake_validity = discriminator(f_foils, f_conds)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
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
            batch_count += 1

            # -----------------
            #  Train Generator
            # -----------------
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z_gen = torch.randn(batch_size, config.get('noise_dimension')).to(device)
                fake_foil = generator(z_gen, conds)
                
                # Evaluate to find F_eps for generator optimization
                _, f_idx_gen = evaluate_physics(fake_foil, conds, norm_stats, current_eps)
                
                if len(f_idx_gen) > 0:
                    f_foil_gen = fake_foil[f_idx_gen]
                    f_conds_gen = conds[f_idx_gen]
                    
                    fake_validity = discriminator(f_foil_gen, f_conds_gen)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()
                    
                    epoch_g_loss += g_loss.item()
                    g_batch_count += 1
                else:
                    # All samples physically reasonable, skip generator update
                    pass
                
        # Calculate average loss for the epoch
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / g_batch_count if g_batch_count > 0 else 0
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]")

    # Save models
    torch.save(generator.state_dict(), "model/generator.pth")
    torch.save(discriminator.state_dict(), "model/discriminator.pth")
    print("Training finished and models saved to model/generator.pth and model/discriminator.pth")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="D Loss")
    plt.plot(g_losses, label="G Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("model/loss_curve.png")
    plt.close()
    print("Loss curve saved to model/loss_curve.png")

if __name__ == "__main__":
    train()