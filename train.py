import yaml
import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from dataset import AirfoilDataset


def compute_gradient_penalty(D, real_samples, fake_samples, conds, device):
    """Calculates the gradient penalty loss for WGAN GP"""
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

def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset & DataLoader
    dataset = AirfoilDataset("model/airfoil_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Initialize models
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    # Optimizers (WGAN-GP uses Adam with beta1=0, beta2=0.9 usually)
    lr = 1e-4
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    epochs = 200
    n_critic = 3
    lambda_gp = 10

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

            # Generate a batch of fake foils
            z = torch.randn(batch_size, config.get('noise_dimension')).to(device)
            fake_foils = generator(z, conds)

            real_validity = discriminator(foils, conds)
            fake_validity = discriminator(fake_foils.detach(), conds)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, foils, fake_foils.detach(), conds, device)

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

                # Generate a batch of foils
                fake_foil = generator(z, conds)
                # Loss measures generator's ability to fool the discriminator
                fake_validity = discriminator(fake_foil, conds)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
                g_batch_count += 1
                
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