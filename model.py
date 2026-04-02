import torch
import torch.nn as nn
import yaml
import math

def center_dense_spacing(M, s_le=0.5, beta=2.0):
    # Generates points in [0, 1] dense ONLY at s_le (Leading Edge)
    M_left = max(2, int(round(M * s_le)))
    M_right = M - M_left + 1
    
    u_left = torch.linspace(0, 1, M_left)
    t_left = s_le * (1.0 - (1.0 - u_left)**beta)
    
    u_right = torch.linspace(0, 1, M_right)[1:]
    t_right = s_le + (1.0 - s_le) * (u_right**beta)
    
    t = torch.cat([t_left, t_right])
    
    # Ensure strict bounds due to float32 precision
    t[0] = 0.0
    t[-1] = 1.0
    t = torch.clamp(t, 0.0, 1.0)
    
    return t

class BezierDecoderLayer(nn.Module):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.num_control_points = self.config['num_control_points']
        self.num_output_points = self.config['num_output_points']
        self.point_density_beta = self.config['point_density_beta']
        
        # Precompute t and Bernstein polynomials
        t = center_dense_spacing(self.num_output_points, s_le=0.5, beta=self.point_density_beta)
        self.register_buffer('t', t)
        
        n = self.num_control_points - 1
        B = torch.zeros((self.num_output_points, n + 1), dtype=torch.float64)
        t_double = t.to(torch.float64)
        
        for i in range(n + 1):
            coeff = math.comb(n, i)
            B[:, i] = coeff * (t_double ** i) * ((1.0 - t_double) ** (n - i))
            
        self.register_buffer('B', B.to(torch.float32))

    def forward(self, control_points, weights):
        """
        control_points: (Batch, N, 2)
        weights: (Batch, N)
        return: (Batch, M, 2)
        """
        # B shape: (M, N)
        weighted_P = control_points * weights.unsqueeze(-1)
        
        # Expand B for batch multiplication
        batch_size = control_points.shape[0]
        B_batch = self.B.unsqueeze(0).expand(batch_size, -1, -1) # (Batch, M, N)
        
        # numerator: (Batch, M, 2)
        numerator = torch.bmm(B_batch, weighted_P)
        
        # denominator: (Batch, M, 1)
        denominator = torch.bmm(B_batch, weights.unsqueeze(-1))
        
        # Avoid division by zero
        curve = numerator / (denominator + 1e-8)
        return curve

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noise_dim = config.get('noise_dimension')
        self.cond_dim = 5
        self.hid_node = config.get('gen_hid_node')
        self.hid_layer = config.get('gen_hid_layer')
        
        act_fun = nn.LeakyReLU(0.2)
        layers = []
        in_dim = self.noise_dim + self.cond_dim
        for _ in range(self.hid_layer):
            layers.append(nn.Linear(in_dim, self.hid_node))
            layers.append(act_fun)
            in_dim = self.hid_node
            
        self.fc_blocks = nn.Sequential(*layers)
        
        self.num_cp = config.get('num_control_points')
        self.out_layer = nn.Linear(self.hid_node, self.num_cp * 3)
        self.bezier_layer = BezierDecoderLayer()

    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        x = self.fc_blocks(x)
        x = self.out_layer(x) # (Batch, N * 3)
        
        # Reshape to control points and weights
        x = x.view(-1, self.num_cp, 3)
        control_points = x[:, :, :2].clone()
        
        # Fix the first and last control points to (-1, 0)
        fixed_pt = torch.tensor([1.0, 0.0], device=x.device, dtype=x.dtype)
        control_points[:, 0, :] = fixed_pt
        control_points[:, -1, :] = fixed_pt
        
        # Weights should be positive to avoid negative denominators / singular curves
        weights = torch.nn.functional.softplus(x[:, :, 2])
        
        curve = self.bezier_layer(control_points, weights) # (Batch, M, 2)
        return curve.view(curve.size(0), -1) # Flatten to (Batch, M*2)

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cond_dim = 5
        self.input_dim = config.get('num_output_points') * 2
        self.hid_node = config.get('gen_hid_node')
        self.hid_layer = config.get('gen_hid_layer')
        
        act_fun = nn.LeakyReLU(0.2)
            
        layers = []
        in_dim = self.input_dim + self.cond_dim
        for _ in range(self.hid_layer):
            layers.append(nn.Linear(in_dim, self.hid_node))
            layers.append(act_fun)
            in_dim = self.hid_node
            
        layers.append(nn.Linear(self.hid_node, 1))
        self.fc_blocks = nn.Sequential(*layers)

    def forward(self, coords, cond):
        x = torch.cat([coords, cond], dim=1)
        validity = self.fc_blocks(x)
        return validity