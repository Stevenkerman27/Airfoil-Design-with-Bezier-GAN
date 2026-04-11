import torch
import yaml
from model import Discriminator

def test_discriminator_forward():
    config = {
        'num_output_points': 100,
        'dis_hid_node': 256,
        'dis_hid_layer': 2,
        'disc_conv_channels': 16,
        'disc_conv_kernel': 15,
        'disc_conv2_kernel': 11,
        'disc_conv2_channels': 16,
        'disc_conv2_stride': 3,
        'cond_dim': 4,
        'gen_hid_fun': 'LeakyRELU'
    }
    
    model = Discriminator(config)
    batch_size = 4
    
    # input coords shape: (batch_size, num_output_points * 2)
    coords = torch.randn(batch_size, 100 * 2)
    cond = torch.randn(batch_size, 4)
    
    output = model(coords, cond)
    
    assert output.shape == (batch_size, 1), f"Expected shape (4, 1), got {output.shape}"

if __name__ == "__main__":
    test_discriminator_forward()
    print("Discriminator forward pass test passed!")