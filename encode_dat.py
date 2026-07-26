import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from artifact_io import save_report_figure, save_yaml
from cst import (
    bounded_cst_exponent,
    build_bernstein_basis,
    decode_split_surface_cst,
    split_surface_t_values,
)
from train_surrogate import load_config, resolve_device


CST_ENCODE_CONFIG_KEY = 'cst_encode'
CST_INPUT_DIRECTORY = 'foildata/processed_foil'
CST_ENCODED_OUTPUT_PATH = 'model/cst_encoded_airfoils.pt'
CST_REPORT_PATH = 'reports/cst/cst_encode_report.yaml'
CST_WORST_CASE_DIRECTORY = 'reports/cst/cst_encode_worst10'


def load_dat(file_path):
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return torch.tensor(points, dtype=torch.float32)


def load_dat_batch(target_paths):
    point_sets = [load_dat(path) for path in target_paths]
    point_count = point_sets[0].shape[0]
    for path, points in zip(target_paths, point_sets):
        if points.shape[0] != point_count:
            raise ValueError(
                f'{path} has {points.shape[0]} points, expected {point_count}'
            )
    return torch.stack(point_sets, dim=0)


def validate_cst_configuration(config):
    if config['cst']['shape_coefficient_count'] < 2:
        raise ValueError(
            'cst.shape_coefficient_count must be at least 2, '
            f"got {config['cst']['shape_coefficient_count']}"
        )
    for name in ['n1_range', 'n2_range']:
        value_range = config['cst'][name]
        if len(value_range) != 2:
            raise ValueError(f'cst.{name} must contain exactly two values')
        lower, upper = value_range
        if lower <= 0.0 or upper <= lower:
            raise ValueError(
                f'cst.{name} must satisfy 0 < lower < upper, got {value_range}'
            )
    encode_config = config[CST_ENCODE_CONFIG_KEY]
    for name in ['batch_size', 'iterations', 'log_every_airfoils', 'worst_case_count']:
        if encode_config[name] <= 0:
            raise ValueError(f'cst_encode.{name} must be positive')


def split_upper_point_count(target_points, point_density_beta):
    upper_t, _ = split_surface_t_values(
        target_points.shape[1],
        point_density_beta,
    )
    expected_le_index = upper_t.shape[0] - 1
    leading_edge_indices = torch.argmin(target_points[:, :, 0], dim=1)
    invalid_indices = torch.nonzero(
        leading_edge_indices != expected_le_index,
        as_tuple=False,
    ).flatten()
    if invalid_indices.numel() > 0:
        sample_index = int(invalid_indices[0].item())
        actual_index = int(leading_edge_indices[sample_index].item())
        raise ValueError(
            'Split-surface data must use the shared leading-edge index '
            f'{expected_le_index}; sample {sample_index} has index {actual_index}. '
            'Re-run foildata/manage_foildata.py before encoding.'
        )
    return upper_t.shape[0]


def initialize_cst_surface_parameters(basis, x_values, target_y, n1, n2):
    class_function = x_values.pow(n1) * (1.0 - x_values).pow(n2)
    design_matrix = torch.cat(
        [class_function.unsqueeze(-1) * basis, x_values.unsqueeze(-1)],
        dim=2,
    )
    return torch.linalg.lstsq(
        design_matrix,
        target_y.unsqueeze(-1),
    ).solution.squeeze(-1)


def fit_cst_airfoil_batch(target_paths, config, device, verbose=False):
    validate_cst_configuration(config)
    cst_config = config['cst']
    encode_config = config[CST_ENCODE_CONFIG_KEY]
    target_points = load_dat_batch(target_paths).to(device)
    if target_points.shape[1] != config['num_output_points']:
        raise ValueError(
            f"Batch has {target_points.shape[1]} points per airfoil, expected "
            f"{config['num_output_points']}"
        )

    upper_point_count = split_upper_point_count(
        target_points,
        config['point_density_beta'],
    )
    upper_x = target_points[:, :upper_point_count, 0]
    lower_x = target_points[:, upper_point_count - 1:, 0]
    coefficient_count = cst_config['shape_coefficient_count']
    upper_basis = build_bernstein_basis(upper_x, coefficient_count)
    lower_basis = build_bernstein_basis(lower_x, coefficient_count)
    initial_n1 = torch.as_tensor(
        sum(cst_config['n1_range']) / 2.0,
        device=device,
        dtype=target_points.dtype,
    )
    initial_n2 = torch.as_tensor(
        sum(cst_config['n2_range']) / 2.0,
        device=device,
        dtype=target_points.dtype,
    )
    upper_initial = initialize_cst_surface_parameters(
        upper_basis,
        upper_x,
        target_points[:, :upper_point_count, 1],
        initial_n1,
        initial_n2,
    )
    lower_initial = initialize_cst_surface_parameters(
        lower_basis,
        lower_x,
        target_points[:, upper_point_count - 1:, 1],
        initial_n1,
        initial_n2,
    )

    upper_coefficients = torch.nn.Parameter(upper_initial[:, :-1])
    lower_coefficients = torch.nn.Parameter(lower_initial[:, :-1])
    upper_te_y = torch.nn.Parameter(upper_initial[:, -1:])
    lower_te_y = torch.nn.Parameter(lower_initial[:, -1:])
    n1_logits = torch.nn.Parameter(torch.zeros_like(upper_te_y))
    n2_logits = torch.nn.Parameter(torch.zeros_like(lower_te_y))
    optimizer = optim.Adam(
        [
            upper_coefficients,
            lower_coefficients,
            upper_te_y,
            lower_te_y,
            n1_logits,
            n2_logits,
        ],
        lr=encode_config['lr'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=encode_config['scheduler_patience'],
        factor=encode_config['scheduler_factor'],
    )

    for iteration in range(encode_config['iterations']):
        optimizer.zero_grad()
        n1 = bounded_cst_exponent(n1_logits, cst_config['n1_range'])
        n2 = bounded_cst_exponent(n2_logits, cst_config['n2_range'])
        curve = decode_split_surface_cst(
            upper_basis,
            lower_basis,
            upper_x,
            lower_x,
            upper_coefficients,
            lower_coefficients,
            upper_te_y,
            lower_te_y,
            n1,
            n2,
        )
        mae_y_per_airfoil = torch.mean(
            torch.abs(curve[:, :, 1] - target_points[:, :, 1]),
            dim=1,
        )
        fit_loss = torch.sum(mae_y_per_airfoil) * encode_config['loss_scale']
        coefficient_reg = (
            torch.mean(upper_coefficients.square())
            + torch.mean(lower_coefficients.square())
        ) * encode_config['coefficient_reg']
        total_loss = fit_loss + coefficient_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        if verbose and (
            iteration % encode_config['log_every_airfoils'] == 0
            or iteration == encode_config['iterations'] - 1
        ):
            print(
                f'CST iteration {iteration}: total={total_loss.item():.8f}, '
                f'mae_y={torch.mean(mae_y_per_airfoil).item():.8f}'
            )

    with torch.no_grad():
        n1 = bounded_cst_exponent(n1_logits, cst_config['n1_range'])
        n2 = bounded_cst_exponent(n2_logits, cst_config['n2_range'])
        curve = decode_split_surface_cst(
            upper_basis,
            lower_basis,
            upper_x,
            lower_x,
            upper_coefficients,
            lower_coefficients,
            upper_te_y,
            lower_te_y,
            n1,
            n2,
        )
        y_difference = curve[:, :, 1] - target_points[:, :, 1]
        mae_y_per_airfoil = torch.mean(torch.abs(y_difference), dim=1)
        mse_y_per_airfoil = torch.mean(y_difference.square(), dim=1)
        max_point_error_per_airfoil = torch.linalg.vector_norm(
            curve - target_points,
            dim=2,
        ).max(dim=1).values

    return {
        'parameters': {
            'upper_coefficients': upper_coefficients.detach().cpu(),
            'lower_coefficients': lower_coefficients.detach().cpu(),
            'upper_te_y': upper_te_y.detach().cpu(),
            'lower_te_y': lower_te_y.detach().cpu(),
            'n1': n1.detach().cpu(),
            'n2': n2.detach().cpu(),
        },
        'curve': curve.detach().cpu(),
        'target_points': target_points.detach().cpu(),
        'mae_y_per_airfoil': mae_y_per_airfoil.detach().cpu(),
        'mse_y_per_airfoil': mse_y_per_airfoil.detach().cpu(),
        'max_point_error_per_airfoil': max_point_error_per_airfoil.detach().cpu(),
    }


def metric_summary(values):
    return {
        'mean': float(torch.mean(values).item()),
        'median': float(torch.median(values).item()),
        'p95': float(torch.quantile(values, 0.95).item()),
        'max': float(torch.max(values).item()),
    }


def visualize_cst_result(target_points, curve_points, airfoil_name, mae_y, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(target_points[:, 0], target_points[:, 1], 'k.', label='Original', markersize=3)
    plt.plot(
        curve_points[:, 0],
        curve_points[:, 1],
        'r-',
        label='CST reconstruction',
        linewidth=1.5,
    )
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f'{airfoil_name} CST reconstruction, MAE_y={mae_y:.8f}')
    save_report_figure(plt.gcf(), save_path, dpi=200, bbox_inches='tight')
    plt.close()


def run_cst_encoding(config_path):
    config = load_config(config_path)
    validate_cst_configuration(config)
    encode_config = config[CST_ENCODE_CONFIG_KEY]
    airfoil_paths = sorted(Path(CST_INPUT_DIRECTORY).glob('*.dat'))
    if len(airfoil_paths) == 0:
        raise ValueError(f'No .dat files found in {CST_INPUT_DIRECTORY}')
    device = resolve_device(config)
    batch_results = []
    for start in range(0, len(airfoil_paths), encode_config['batch_size']):
        batch_paths = airfoil_paths[start:start + encode_config['batch_size']]
        batch_results.append(
            fit_cst_airfoil_batch(
                [str(path) for path in batch_paths],
                config,
                device,
            )
        )
        encoded_count = min(start + encode_config['batch_size'], len(airfoil_paths))
        crossed_log_interval = (
            encoded_count // encode_config['log_every_airfoils']
            > start // encode_config['log_every_airfoils']
        )
        if crossed_log_interval or encoded_count == len(airfoil_paths):
            print(f'CST encoded {encoded_count}/{len(airfoil_paths)} airfoils')

    parameters = {
        name: torch.cat([result['parameters'][name] for result in batch_results], dim=0)
        for name in batch_results[0]['parameters']
    }
    all_curves = torch.cat([result['curve'] for result in batch_results], dim=0)
    all_targets = torch.cat([result['target_points'] for result in batch_results], dim=0)
    all_metrics = {
        name: torch.cat([result[name] for result in batch_results], dim=0)
        for name in [
            'mae_y_per_airfoil',
            'mse_y_per_airfoil',
            'max_point_error_per_airfoil',
        ]
    }
    encoded_path = Path(CST_ENCODED_OUTPUT_PATH)
    encoded_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format': 'cst_v2',
            'airfoil_paths': [str(path) for path in airfoil_paths],
            'parameters': parameters,
            'raw_target_points': all_targets,
            'reconstructed_points': all_curves,
            'metrics': all_metrics,
        },
        encoded_path,
    )

    ranked_indices = torch.argsort(all_metrics['mae_y_per_airfoil'], descending=True)
    worst_count = min(encode_config['worst_case_count'], len(airfoil_paths))
    worst_dir = Path(CST_WORST_CASE_DIRECTORY)
    if worst_dir.exists():
        shutil.rmtree(worst_dir)
    worst_dir.mkdir(parents=True, exist_ok=True)
    worst_airfoils = []
    for rank, tensor_index in enumerate(ranked_indices[:worst_count], start=1):
        index = int(tensor_index.item())
        source_path = airfoil_paths[index]
        plot_path = worst_dir / f'{rank:02d}_{source_path.stem}_comparison.png'
        mae_y = float(all_metrics['mae_y_per_airfoil'][index].item())
        visualize_cst_result(
            all_targets[index],
            all_curves[index],
            source_path.name,
            mae_y,
            plot_path,
        )
        worst_airfoils.append(
            {
                'rank': rank,
                'name': source_path.name,
                'mae_y': mae_y,
                'mse_y': float(all_metrics['mse_y_per_airfoil'][index].item()),
                'max_point_error': float(all_metrics['max_point_error_per_airfoil'][index].item()),
                'comparison_plot_path': str(plot_path),
            }
        )

    report = {
        'format': 'cst_encode_report_v2',
        'airfoil_count': len(airfoil_paths),
        'cst': config['cst'],
        'metrics': {
            'mae_y': metric_summary(all_metrics['mae_y_per_airfoil']),
            'mse_y': metric_summary(all_metrics['mse_y_per_airfoil']),
            'max_point_error': metric_summary(all_metrics['max_point_error_per_airfoil']),
        },
        'worst_airfoils': worst_airfoils,
    }
    report_path = Path(CST_REPORT_PATH)
    save_yaml(str(report_path), report)
    print(f'CST parameters and reconstructions saved to {encoded_path}')
    print(f'CST report saved to {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Batch encode processed airfoils with configurable-order CST curves'
    )
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    run_cst_encoding(args.config)


if __name__ == '__main__':
    main()
