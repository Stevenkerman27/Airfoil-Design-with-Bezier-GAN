import torch
import torch.nn as nn
from cst import (
    build_bernstein_basis,
    bounded_cst_exponent,
    decode_split_surface_cst,
    split_surface_t_values,
)

class CSTDecoderLayer(nn.Module):
    def __init__(self, shape_coefficient_count, num_output_points, point_density_beta):
        super().__init__()
        if shape_coefficient_count < 2:
            raise ValueError(
                'cst.shape_coefficient_count must be at least 2, '
                f'got {shape_coefficient_count}'
            )
        if num_output_points < 3:
            raise ValueError(
                f'num_output_points must be at least 3, got {num_output_points}'
            )

        upper_t, lower_t = split_surface_t_values(
            num_output_points,
            point_density_beta,
        )
        upper_x = 1.0 - upper_t
        lower_x = lower_t
        self.register_buffer('upper_x', upper_x)
        self.register_buffer('lower_x', lower_x)
        self.register_buffer(
            'upper_basis',
            build_bernstein_basis(upper_x, shape_coefficient_count),
        )
        self.register_buffer(
            'lower_basis',
            build_bernstein_basis(lower_x, shape_coefficient_count),
        )

    def forward(
        self,
        upper_coefficients,
        lower_coefficients,
        upper_te_y,
        lower_te_y,
        n1,
        n2,
    ):
        return decode_split_surface_cst(
            self.upper_basis,
            self.lower_basis,
            self.upper_x,
            self.lower_x,
            upper_coefficients,
            lower_coefficients,
            upper_te_y,
            lower_te_y,
            n1,
            n2,
        )

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noise_dim = config['noise_dimension']
        self.cond_dim = config['cond_dim']
        self.hid_node = config['gen_hid_node']
        self.hid_layer = config['gen_hid_layer']
        
        act_fun = nn.LeakyReLU(0.2)
            
        layers = []
        in_dim = self.noise_dim + self.cond_dim
        for _ in range(self.hid_layer):
            layers.append(nn.Linear(in_dim, self.hid_node))
            layers.append(act_fun)
            in_dim = self.hid_node
            
        self.fc_blocks = nn.Sequential(*layers)
        
        self.shape_coefficient_count = config['cst']['shape_coefficient_count']
        self.n1_range = config['cst']['n1_range']
        self.n2_range = config['cst']['n2_range']
        self._validate_cst_config()
        self.parameter_dimension = 2 * self.shape_coefficient_count + 4
        self.out_layer = nn.Linear(self.hid_node, self.parameter_dimension)
        self.cst_layer = CSTDecoderLayer(
            self.shape_coefficient_count,
            config['num_output_points'],
            config['point_density_beta'],
        )

        coord_norm = torch.load("model/coord_norm.pt", map_location='cpu', weights_only=True)
        self.register_buffer('coord_x_min', torch.as_tensor(coord_norm['x_min'], dtype=torch.float32))
        self.register_buffer('coord_x_max', torch.as_tensor(coord_norm['x_max'], dtype=torch.float32))
        self.register_buffer('coord_y_min', torch.as_tensor(coord_norm['y_min'], dtype=torch.float32))
        self.register_buffer('coord_y_max', torch.as_tensor(coord_norm['y_max'], dtype=torch.float32))

    def _validate_cst_config(self):
        if self.shape_coefficient_count < 2:
            raise ValueError(
                'cst.shape_coefficient_count must be at least 2, '
                f'got {self.shape_coefficient_count}'
            )
        for name, value_range in [('n1_range', self.n1_range), ('n2_range', self.n2_range)]:
            if len(value_range) != 2:
                raise ValueError(f'cst.{name} must contain exactly two values')
            lower, upper = value_range
            if lower <= 0.0 or upper <= lower:
                raise ValueError(
                    f'cst.{name} must satisfy 0 < lower < upper, got {value_range}'
                )

    @staticmethod
    def _bounded_parameter(raw_value, value_range):
        return bounded_cst_exponent(raw_value, value_range)

    def decode_parameters(self, parameters):
        offset = 0

        upper_coefficients = parameters[:, offset:offset + self.shape_coefficient_count]
        offset += self.shape_coefficient_count
        lower_coefficients = parameters[:, offset:offset + self.shape_coefficient_count]
        offset += self.shape_coefficient_count
        upper_te_y = parameters[:, offset:offset + 1]
        offset += 1
        lower_te_y = parameters[:, offset:offset + 1]
        offset += 1
        n1 = self._bounded_parameter(parameters[:, offset:offset + 1], self.n1_range)
        offset += 1
        n2 = self._bounded_parameter(parameters[:, offset:offset + 1], self.n2_range)
        offset += 1
        if offset != self.parameter_dimension:
            raise RuntimeError(
                f'Expected {self.parameter_dimension} generator parameters, consumed {offset}'
            )
        return (
            upper_coefficients,
            lower_coefficients,
            upper_te_y,
            lower_te_y,
            n1,
            n2,
        )

    def normalize_coordinates(self, physical_coordinates):
        x_range = self.coord_x_max - self.coord_x_min
        y_range = self.coord_y_max - self.coord_y_min
        if x_range <= 0.0 or y_range <= 0.0:
            raise ValueError('model/coord_norm.pt must have positive x and y ranges')
        normalized_x = (physical_coordinates[..., 0] - self.coord_x_min) / x_range
        normalized_y = (physical_coordinates[..., 1] - self.coord_y_min) / y_range
        return torch.stack([normalized_x, normalized_y], dim=-1)

    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        x = self.fc_blocks(x)
        parameters = self.out_layer(x)
        decoded_parameters = self.decode_parameters(parameters)
        physical_curve = self.cst_layer(
            *decoded_parameters,
        )
        curve = self.normalize_coordinates(physical_curve)
        return curve.view(curve.size(0), -1) # Flatten to (Batch, M*2)

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cond_dim = config['cond_dim']
        self.num_pts = config['num_output_points']
        self.input_dim = self.num_pts * 2
        self.hid_node = config['dis_hid_node']
        self.hid_layer = config['dis_hid_layer']
        
        # Conv layer parameters
        self.conv_channels = config['disc_conv_channels']
        self.kernel_size = config['disc_conv_kernel']
        
        self.conv2_kernel = config['disc_conv2_kernel']
        self.conv2_channels = config['disc_conv2_channels']
        self.conv2_stride = config['disc_conv2_stride']
        
        # Stage 1: Convolutional Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=2, 
                               out_channels=self.conv_channels, 
                               kernel_size=self.kernel_size, 
                               padding=self.kernel_size // 2)
        
        self.conv2 = nn.Conv1d(in_channels=self.conv_channels,
                               out_channels=self.conv2_channels,
                               kernel_size=self.conv2_kernel,
                               stride=self.conv2_stride,
                               padding=self.conv2_kernel // 2)
        
        act_fun = nn.LeakyReLU(0.2)
            
        # Calculate sequence length after conv2
        # conv1 output length is num_pts (due to padding = kernel // 2, stride = 1)
        seq_len = (self.num_pts + 2 * (self.conv2_kernel // 2) - self.conv2_kernel) // self.conv2_stride + 1
        
        layers = []
        # First FC layer input = (conv2_channels * seq_len) + cond_dim
        in_dim = (self.conv2_channels * seq_len) + self.cond_dim
        
        # Shared hidden layers
        for _ in range(self.hid_layer - 1):
            layers.append(nn.Linear(in_dim, self.hid_node))
            layers.append(act_fun)
            in_dim = self.hid_node
            
        self.shared_fc = nn.Sequential(*layers)
        
        self.adv_layer = nn.Linear(in_dim, 1)

    def forward(self, coords, cond):
        # coords: (Batch, M*2) -> (Batch, M, 2) -> (Batch, 2, M)
        batch_size = coords.size(0)
        x = coords.view(batch_size, self.num_pts, 2).permute(0, 2, 1)
        
        # Conv1 + Activation
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        
        # Conv2 + Activation
        x = torch.nn.functional.leaky_relu(self.conv2(x), 0.2)
        
        # Flatten: (Batch, out_channels, seq_len) -> (Batch, out_channels * seq_len)
        x = x.view(batch_size, -1)
        
        # Concat with conditions
        x = torch.cat([x, cond], dim=1)
        
        # Shared FC blocks
        features = self.shared_fc(x)
        
        # Outputs
        validity = self.adv_layer(features)
        return validity

class AerodynamicSurrogate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cond_dim = config['surrogate_cond_dim']
        self.out_dim = config['surrogate_out_dim']
        self.num_pts = config['num_output_points']
        self.hid_node = config['surrogate_hid_node']
        self.hid_layer = config['surrogate_hid_layer']

        self.conv_channels = config['surrogate_conv1_channels']
        self.kernel_size = config['surrogate_conv1_kernel']
        self.conv2_kernel = config['surrogate_conv2_kernel']
        self.conv2_channels = config['surrogate_conv2_channels']
        self.conv2_stride = config['surrogate_conv2_stride']

        for name, kernel_size in (
            ('surrogate_conv1_kernel', self.kernel_size),
            ('surrogate_conv2_kernel', self.conv2_kernel),
        ):
            if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f'{name} must be a positive odd integer, got {kernel_size}')
        if not isinstance(self.conv2_stride, int) or self.conv2_stride <= 0:
            raise ValueError(
                f'surrogate_conv2_stride must be a positive integer, got {self.conv2_stride}'
            )

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv2_channels,
            kernel_size=self.conv2_kernel,
            stride=self.conv2_stride,
            padding=self.conv2_kernel // 2,
        )

        conv1_len = self.num_pts + 2 * (self.kernel_size // 2) - self.kernel_size + 1
        seq_len = (
            (conv1_len + 2 * (self.conv2_kernel // 2) - self.conv2_kernel)
            // self.conv2_stride
            + 1
        )
        if seq_len <= 0:
            raise ValueError(
                'Surrogate convolution configuration produces a non-positive sequence length: '
                f'{seq_len}'
            )
        in_dim = (self.conv2_channels * seq_len) + self.cond_dim

        act_fun = nn.LeakyReLU(0.2)
        layers = []
        for _ in range(self.hid_layer):
            layers.append(nn.Linear(in_dim, self.hid_node))
            layers.append(act_fun)
            in_dim = self.hid_node

        self.fc_blocks = nn.Sequential(*layers)
        self.out_layer = nn.Linear(in_dim, self.out_dim)

    def forward(self, coords, conditions):
        batch_size = coords.size(0)
        x = coords.view(batch_size, self.num_pts, 2).permute(0, 2, 1)
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = x.view(batch_size, -1)
        x = torch.cat([x, conditions], dim=1)
        x = self.fc_blocks(x)
        return self.out_layer(x)
