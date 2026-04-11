import torch
import yaml
import matplotlib.pyplot as plt
import os
from model import Discriminator

def visualize_discriminator_conv_weights():
    # 1. Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. Load state dict first to determine architecture if possible
    device = torch.device('cpu')
    model_path = 'model/discriminator.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Update config from state_dict to match architecture
    if 'fc_blocks.0.weight' in state_dict:
        # dis_hid_node is the output of the first FC layer
        config['dis_hid_node'] = state_dict['fc_blocks.0.weight'].shape[0]
        print(f"Detected dis_hid_node = {config['dis_hid_node']} from weights.")
    
    # Ensure conv parameters are detected from state_dict as well
    if 'conv1.weight' in state_dict:
        config['disc_conv_channels'] = state_dict['conv1.weight'].shape[0]
        config['disc_conv_kernel'] = state_dict['conv1.weight'].shape[2]
        print(f"Detected conv1: channels={config['disc_conv_channels']}, kernel={config['disc_conv_kernel']}")
    
    if 'conv2.weight' in state_dict:
        config['disc_conv2_channels'] = state_dict['conv2.weight'].shape[0]
        config['disc_conv2_kernel'] = state_dict['conv2.weight'].shape[2]
        print(f"Detected conv2: channels={config['disc_conv2_channels']}, kernel={config['disc_conv2_kernel']}")

    discriminator = Discriminator(config).to(device)
    discriminator.load_state_dict(state_dict)
    
    # 3. Extract weights
    weights1 = discriminator.conv1.weight.detach().numpy() # (C_out1, 2, K1)
    weights2 = discriminator.conv2.weight.detach().numpy() # (C_out2, C_in2, K2)
    
    # 4. Plot Layer 1 (conv1)
    out_channels1 = weights1.shape[0]
    kernel_size1 = weights1.shape[2]
    cols1 = 4
    rows1 = (out_channels1 + cols1 - 1) // cols1
    
    fig1, axes1 = plt.subplots(rows1, cols1, figsize=(15, 3 * rows1))
    fig1.suptitle('Discriminator Conv Layer 1 Weights (X/Y)', fontsize=16)
    axes1 = axes1.flatten()
    
    x1 = list(range(kernel_size1))
    for i in range(out_channels1):
        ax = axes1[i]
        ax.plot(x1, weights1[i, 0, :], 'r-o', label='X Weight')
        ax.plot(x1, weights1[i, 1, :], 'b-x', label='Y Weight')
        ax.set_title(f'Filter {i}')
        ax.grid(True)
        if i == 0:
            ax.legend()
    for i in range(out_channels1, len(axes1)):
        axes1[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model/discriminator_conv1_weights.png')
    print("Saved Layer 1 plot to model/discriminator_conv1_weights.png")

    # 5. Plot Layer 2 (conv2)
    out_channels2 = weights2.shape[0]
    in_channels2 = weights2.shape[1]
    kernel_size2 = weights2.shape[2]
    
    # Conv2 has in_channels2 (e.g. 16). Heatmap is better.
    cols2 = 4
    rows2 = (out_channels2 + cols2 - 1) // cols2
    
    fig2, axes2 = plt.subplots(rows2, cols2, figsize=(15, 4 * rows2))
    fig2.suptitle('Discriminator Conv Layer 2 Weights (Heatmaps)', fontsize=16)
    axes2 = axes2.flatten()
    
    for i in range(out_channels2):
        ax = axes2[i]
        # weights2[i] shape: (in_channels2, kernel_size2)
        im = ax.imshow(weights2[i], aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Filter {i}')
        ax.set_xlabel('Kernel Pos')
        ax.set_ylabel('In Channel')
        fig2.colorbar(im, ax=ax)
        
    for i in range(out_channels2, len(axes2)):
        axes2[i].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model/discriminator_conv2_weights.png')
    print("Saved Layer 2 plot to model/discriminator_conv2_weights.png")

if __name__ == '__main__':
    visualize_discriminator_conv_weights()
