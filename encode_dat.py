import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from model import BezierDecoderLayer
from train_surrogate import load_config, resolve_device


BEZIER_ENCODE_CONFIG_KEY = 'bezier_encode'


def scalar_on_device(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def load_dat(file_path):
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])
                except ValueError:
                    pass
    return torch.tensor(points, dtype=torch.float32)


def load_dat_batch(target_paths):
    raw_points = []
    expected_points = None
    for target_path in target_paths:
        points = load_dat(target_path)
        if expected_points is None:
            expected_points = points.shape[0]
        if points.shape[0] != expected_points:
            raise ValueError(
                f"{target_path} has {points.shape[0]} points, expected {expected_points}"
            )
        raw_points.append(points)
    return torch.stack(raw_points, dim=0)


def load_coord_norm(coord_norm_path):
    if not os.path.exists(coord_norm_path):
        raise FileNotFoundError(f"Coordinate normalization file not found: {coord_norm_path}")
    return torch.load(coord_norm_path, map_location='cpu', weights_only=True)


def normalize_points(raw_points, coord_norm):
    x_min = scalar_on_device(coord_norm['x_min'], raw_points.device, raw_points.dtype)
    x_max = scalar_on_device(coord_norm['x_max'], raw_points.device, raw_points.dtype)
    y_min = scalar_on_device(coord_norm['y_min'], raw_points.device, raw_points.dtype)
    y_max = scalar_on_device(coord_norm['y_max'], raw_points.device, raw_points.dtype)
    points = raw_points.clone()
    points[:, :, 0] = (points[:, :, 0] - x_min) / (x_max - x_min + 1e-8)
    points[:, :, 1] = (points[:, :, 1] - y_min) / (y_max - y_min + 1e-8)
    return points


def denormalize_points(points, coord_norm):
    x_min = scalar_on_device(coord_norm['x_min'], points.device, points.dtype)
    x_max = scalar_on_device(coord_norm['x_max'], points.device, points.dtype)
    y_min = scalar_on_device(coord_norm['y_min'], points.device, points.dtype)
    y_max = scalar_on_device(coord_norm['y_max'], points.device, points.dtype)
    denorm = points.clone()
    denorm[:, :, 0] = denorm[:, :, 0] * (x_max - x_min + 1e-8) + x_min
    denorm[:, :, 1] = denorm[:, :, 1] * (y_max - y_min + 1e-8) + y_min
    return denorm


def build_fixed_trailing_edge(coord_norm, device):
    x_min = scalar_on_device(coord_norm['x_min'], device, torch.float32)
    x_max = scalar_on_device(coord_norm['x_max'], device, torch.float32)
    y_min = scalar_on_device(coord_norm['y_min'], device, torch.float32)
    y_max = scalar_on_device(coord_norm['y_max'], device, torch.float32)
    te_x_norm = (1.0 - x_min) / (x_max - x_min + 1e-8)
    te_y_norm = (0.0 - y_min) / (y_max - y_min + 1e-8)
    return torch.stack([te_x_norm, te_y_norm]).view(1, 1, 2)


def leading_edge_mse(curve, target_points, window):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    le_index = int(torch.argmin(target_points[0, :, 0]).item())
    start = max(0, le_index - window)
    end = min(target_points.shape[1], le_index + window + 1)
    return torch.mean((curve[:, start:end, :] - target_points[:, start:end, :]) ** 2)


def leading_edge_mse_per_airfoil(curve, target_points, window):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    values = []
    for sample_index in range(target_points.shape[0]):
        le_index = int(torch.argmin(target_points[sample_index, :, 0]).item())
        start = max(0, le_index - window)
        end = min(target_points.shape[1], le_index + window + 1)
        values.append(
            torch.mean(
                (
                    curve[sample_index:sample_index + 1, start:end, :]
                    - target_points[sample_index:sample_index + 1, start:end, :]
                ) ** 2
            )
        )
    return torch.stack(values)


def leading_edge_mae_per_airfoil(curve, target_points, window):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    values = []
    for sample_index in range(target_points.shape[0]):
        le_index = int(torch.argmin(target_points[sample_index, :, 0]).item())
        start = max(0, le_index - window)
        end = min(target_points.shape[1], le_index + window + 1)
        values.append(
            torch.mean(
                torch.abs(
                    curve[sample_index:sample_index + 1, start:end, :]
                    - target_points[sample_index:sample_index + 1, start:end, :]
                )
            )
        )
    return torch.stack(values)


def cosine_leading_edge_weights(target_points, window, amplitude):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    if amplitude < 0:
        raise ValueError(f"leading_edge_weight_amplitude must be non-negative, got {amplitude}")
    batch_size = target_points.shape[0]
    num_points = target_points.shape[1]
    weights = torch.ones(
        (batch_size, num_points),
        dtype=target_points.dtype,
        device=target_points.device,
    )
    if window == 0 or amplitude == 0:
        return weights

    index_positions = torch.arange(num_points, device=target_points.device)
    for sample_index in range(batch_size):
        le_index = int(torch.argmin(target_points[sample_index, :, 0]).item())
        distance = torch.abs(index_positions - le_index)
        mask = distance <= window
        normalized_distance = distance[mask].to(target_points.dtype) / float(window)
        weights[sample_index, mask] = (
            1.0
            + amplitude
            * 0.5
            * (1.0 + torch.cos(torch.pi * normalized_distance))
        )
    return weights


def weighted_mae_per_airfoil(curve, target_points, weights):
    absolute_error = torch.mean(torch.abs(curve - target_points), dim=2)
    return torch.sum(absolute_error * weights, dim=1) / torch.sum(weights, dim=1)


def fit_bezier_airfoil_batch(target_paths, config, coord_norm, device, verbose=False):
    encode_config = config[BEZIER_ENCODE_CONFIG_KEY]
    raw_target_points = load_dat_batch(target_paths).to(device)
    target_points = normalize_points(raw_target_points, coord_norm).to(device)

    expected_points = config['num_output_points']
    actual_points = target_points.shape[1]
    if actual_points != expected_points:
        raise ValueError(
            f"Batch has {actual_points} points per airfoil, expected {expected_points}"
        )

    batch_size = target_points.shape[0]
    leading_edge_weights = cosine_leading_edge_weights(
        target_points,
        encode_config['leading_edge_window'],
        encode_config['leading_edge_weight_amplitude'],
    )
    num_control_points = config['num_control_points']
    indices = torch.linspace(0, actual_points - 1, num_control_points, device=device).long()
    init_cp = target_points[:, indices, :].clone()

    fixed_pt = build_fixed_trailing_edge(coord_norm, device).expand(batch_size, -1, -1)
    trainable_control_points = torch.nn.Parameter(init_cp[:, 1:-1, :])
    weights = torch.nn.Parameter(
        torch.ones((batch_size, num_control_points), dtype=torch.float32, device=device)
    )

    decoder = BezierDecoderLayer(config).to(device)
    optimizer = optim.Adam(
        [trainable_control_points, weights],
        lr=encode_config['lr'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=encode_config['scheduler_patience'],
        factor=encode_config['scheduler_factor'],
    )

    for i in range(encode_config['iterations']):
        optimizer.zero_grad()

        full_control_points = torch.cat([fixed_pt, trainable_control_points, fixed_pt], dim=1)
        abs_weights = torch.abs(weights)
        curve = decoder(full_control_points, abs_weights)

        mae_per_airfoil = weighted_mae_per_airfoil(curve, target_points, leading_edge_weights)
        fit_loss = torch.sum(mae_per_airfoil) * encode_config['loss_scale']
        reg_loss = torch.sum(torch.mean(abs_weights ** 2, dim=1)) * encode_config['weight_reg']

        cp_diff = full_control_points[:, 1:, :] - full_control_points[:, :-1, :]
        length_penalty = (
            torch.sum(torch.mean(cp_diff ** 2, dim=(1, 2)))
            * encode_config['length_penalty']
        )

        total_loss = fit_loss + reg_loss + length_penalty
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        if verbose and (i % encode_config['log_interval'] == 0 or i == encode_config['iterations'] - 1):
            print(
                f"Iter {i}, Total: {total_loss.item():.5f}, "
                f"Weighted MAE: {fit_loss.item():.5f}, Reg: {reg_loss.item():.5f}, "
                f"Len: {length_penalty.item():.5f}"
            )

    with torch.no_grad():
        final_control_points = torch.cat([fixed_pt, trainable_control_points, fixed_pt], dim=1)
        final_weights = torch.abs(weights)
        final_curve = decoder(final_control_points, final_weights)
        final_mae_per_airfoil = torch.mean(torch.abs(final_curve - target_points), dim=(1, 2))
        final_mse_per_airfoil = torch.mean((final_curve - target_points) ** 2, dim=(1, 2))
        max_point_error_per_airfoil = torch.sqrt(
            torch.sum((final_curve - target_points) ** 2, dim=2)
        ).max(dim=1).values
        le_mae_per_airfoil = leading_edge_mae_per_airfoil(
            final_curve,
            target_points,
            encode_config['leading_edge_window'],
        )
        le_mse_per_airfoil = leading_edge_mse_per_airfoil(
            final_curve,
            target_points,
            encode_config['leading_edge_window'],
        )

    return {
        'control_points': final_control_points.detach().cpu(),
        'weights': final_weights.detach().cpu(),
        'curve': final_curve.detach().cpu(),
        'raw_target_points': raw_target_points.detach().cpu(),
        'mae_per_airfoil': final_mae_per_airfoil.detach().cpu(),
        'mse_per_airfoil': final_mse_per_airfoil.detach().cpu(),
        'max_point_error_per_airfoil': max_point_error_per_airfoil.detach().cpu(),
        'leading_edge_mae_per_airfoil': le_mae_per_airfoil.detach().cpu(),
        'leading_edge_mse_per_airfoil': le_mse_per_airfoil.detach().cpu(),
        'mae': float(torch.mean(final_mae_per_airfoil).detach().cpu().item()),
        'mse': float(torch.mean(final_mse_per_airfoil).detach().cpu().item()),
        'max_point_error': float(torch.max(max_point_error_per_airfoil).detach().cpu().item()),
        'leading_edge_mae': float(torch.mean(le_mae_per_airfoil).detach().cpu().item()),
        'leading_edge_mse': float(torch.mean(le_mse_per_airfoil).detach().cpu().item()),
    }


def fit_bezier_to_airfoil(target_path, config, coord_norm, device, verbose=False):
    batch_result = fit_bezier_airfoil_batch(
        [target_path],
        config,
        coord_norm,
        device,
        verbose=verbose,
    )
    return {
        'control_points': batch_result['control_points'],
        'weights': batch_result['weights'],
        'curve': batch_result['curve'],
        'raw_target_points': batch_result['raw_target_points'],
        'mae': batch_result['mae'],
        'mse': batch_result['mse'],
        'max_point_error': batch_result['max_point_error'],
        'leading_edge_mae': batch_result['leading_edge_mae'],
        'leading_edge_mse': batch_result['leading_edge_mse'],
    }


def visualize_result(target_points, curve_points, control_points, weights, config, save_path='model/bezier_fit_result.png'):
    """
    可视化编码结果，并标注控制点权重
    """
    target_np = target_points.squeeze(0).cpu().numpy()
    curve_np = curve_points.squeeze(0).detach().cpu().numpy()
    cp_np = control_points.squeeze(0).detach().cpu().numpy()
    weights_np = weights.squeeze(0).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(target_np[:, 0], target_np[:, 1], 'k.', label='Original .dat', markersize=3)
    plt.plot(curve_np[:, 0], curve_np[:, 1], 'r-', label='Bezier Curve', linewidth=2)
    plt.plot(cp_np[:, 0], cp_np[:, 1], 'gx--', label='Control Points', alpha=0.6, markersize=5)
    
    # 标注每个控制点的权重
    for i, (x, y) in enumerate(cp_np):
        plt.text(x, y + 0.01, f'w={weights_np[i]:.2f}', fontsize=8, color='green', 
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.axis('equal')
    plt.legend()
    plt.title(f"Airfoil Bezier Encoding ({config['num_control_points']} CP, {config['num_output_points']} Pts)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    config = load_config('config.yaml')
    encode_config = config[BEZIER_ENCODE_CONFIG_KEY]
    target_path = encode_config['target_path']
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target airfoil file not found: {target_path}")

    device = resolve_device(config)
    coord_norm = load_coord_norm(encode_config['coord_norm_path'])

    print(f"Starting optimization for {target_path} (Normalized Space)...")
    result = fit_bezier_to_airfoil(target_path, config, coord_norm, device, verbose=True)

    out_file = encode_config['output_path']
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    torch.save({
        'control_points': result['control_points'],
        'weights': result['weights'],
    }, out_file)
    print(f"Encoded parameters saved to {out_file}")

    vis_target = result['raw_target_points']
    vis_curve = denormalize_points(result['curve'], coord_norm)
    vis_cp = denormalize_points(result['control_points'], coord_norm)

    visualize_result(
        vis_target,
        vis_curve,
        vis_cp,
        result['weights'],
        config,
        save_path=encode_config['plot_path'],
    )
    print(
        f"Final MAE: {result['mae']:.8f}, "
        f"Final MSE: {result['mse']:.8f}, "
        f"Leading-edge MAE: {result['leading_edge_mae']:.8f}, "
        f"Leading-edge MSE: {result['leading_edge_mse']:.8f}, "
        f"Max point error: {result['max_point_error']:.8f}"
    )

if __name__ == '__main__':
    main()
