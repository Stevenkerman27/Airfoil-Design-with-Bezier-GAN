import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml

from artifact_io import save_report_figure, save_yaml
from model import Discriminator, Generator
from surrogate_split import load_cross_validation_manifest
from train import normalize_gan_coords


DATASET_PATH = 'model/airfoil_dataset.pt'
COORD_NORM_PATH = 'model/coord_norm.pt'
CONDITION_NORM_PATH = 'model/cond_norm.pt'
DEFAULT_CHECKPOINT_PATH = 'model/gan_final.pt'
DEFAULT_OUTPUT_DIRECTORY = 'reports/gan/discriminator_sensitivity'


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def resolve_device(requested_device):
    if requested_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        return torch.device('cuda')
    if requested_device == 'cpu':
        return torch.device('cpu')
    raise ValueError(f"device must be one of 'auto', 'cuda', or 'cpu', got {requested_device!r}")


def load_models(config, checkpoint_path, device, include_generator):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required_keys = {'discriminator_state_dict'}
    if include_generator:
        required_keys.add('generator_state_dict')
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError(
            f'Checkpoint {checkpoint_path} must contain keys {sorted(required_keys)}'
        )

    discriminator = Discriminator(config).to(device)
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    generator = None
    if include_generator:
        generator = Generator(config).to(device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
    return discriminator, generator


def load_real_development_samples(config, count, seed, device):
    raw_data = torch.load(DATASET_PATH, map_location='cpu', weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    development_indices = manifest['development_indices']
    if count <= 0:
        raise ValueError(f'sample-count must be positive, got {count}')
    if count > len(development_indices):
        raise ValueError(
            f'Requested {count} development samples, only {len(development_indices)} are available'
        )

    random_generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(len(development_indices), generator=random_generator)[:count]
    dataset_indices = [development_indices[position] for position in positions.tolist()]
    physical_coords = torch.stack(
        [raw_data[index]['x'].float().view(-1, 2) for index in dataset_indices]
    ).to(device)
    physical_conditions = torch.stack(
        [raw_data[index]['y'].float() for index in dataset_indices]
    ).to(device)
    return physical_coords, physical_conditions, dataset_indices


def load_generated_samples(config, generator, count, labels, seed, condition_stats, device):
    if count <= 0:
        raise ValueError(f'sample-count must be positive, got {count}')
    if labels is None:
        raise ValueError('--labels is required when --source generated')
    if len(labels) != config['cond_dim']:
        raise ValueError(
            f'--labels must contain {config["cond_dim"]} values in [alpha, Re, CL, CM] order'
        )

    noise_generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        count,
        config['noise_dimension'],
        generator=noise_generator,
        device=device,
    )
    physical_conditions = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(0)
    physical_conditions = physical_conditions.expand(count, -1)
    normalized_conditions = (
        physical_conditions - condition_stats['mean']
    ) / condition_stats['std']
    with torch.no_grad():
        normalized_coords = generator(noise, normalized_conditions).view(
            count, config['num_output_points'], 2
        )
    return normalized_coords, physical_conditions, list(range(count))


def calculate_input_gradients(discriminator, normalized_coords, normalized_conditions):
    critic_input = normalized_coords.detach().clone().requires_grad_(True)
    scores = discriminator(critic_input.view(critic_input.size(0), -1), normalized_conditions)
    gradients = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=critic_input,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    return scores.detach(), gradients.detach()


def split_surface_names(physical_coords):
    leading_edge_index = int(torch.argmin(physical_coords[:, 0]).item())
    surface_names = []
    for point_index in range(physical_coords.size(0)):
        if point_index < leading_edge_index:
            surface_names.append('upper')
        elif point_index == leading_edge_index:
            surface_names.append('leading_edge')
        else:
            surface_names.append('lower')
    return surface_names


def write_sample_csv(path, source, sample_index, dataset_index, physical_coords,
                     normalized_gradients, physical_gradients, score):
    surface_names = split_surface_names(physical_coords)
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'source', 'sample_index', 'dataset_index', 'point_index', 'surface',
                'x_physical', 'y_physical',
                'd_score_d_x_normalized', 'd_score_d_y_normalized',
                'sensitivity_normalized_l2',
                'd_score_d_x_physical', 'd_score_d_y_physical',
                'sensitivity_physical_l2', 'critic_score',
            ],
        )
        writer.writeheader()
        for point_index in range(physical_coords.shape[0]):
            normalized_gradient = normalized_gradients[point_index]
            physical_gradient = physical_gradients[point_index]
            writer.writerow({
                'source': source,
                'sample_index': sample_index,
                'dataset_index': dataset_index,
                'point_index': point_index,
                'surface': surface_names[point_index],
                'x_physical': float(physical_coords[point_index, 0]),
                'y_physical': float(physical_coords[point_index, 1]),
                'd_score_d_x_normalized': float(normalized_gradient[0]),
                'd_score_d_y_normalized': float(normalized_gradient[1]),
                'sensitivity_normalized_l2': float(torch.linalg.vector_norm(normalized_gradient)),
                'd_score_d_x_physical': float(physical_gradient[0]),
                'd_score_d_y_physical': float(physical_gradient[1]),
                'sensitivity_physical_l2': float(torch.linalg.vector_norm(physical_gradient)),
                'critic_score': float(score),
            })


def plot_sample(path, source, sample_index, dataset_index, physical_coords,
                normalized_gradients, physical_gradients, score):
    coords = physical_coords.cpu().numpy()
    normalized_gradient_values = normalized_gradients.cpu().numpy()
    physical_gradient_values = physical_gradients.cpu().numpy()
    physical_sensitivity = np.linalg.vector_norm(physical_gradient_values, axis=1)
    normalized_sensitivity = np.linalg.vector_norm(normalized_gradient_values, axis=1)
    point_indices = np.arange(coords.shape[0])

    figure, axes = plt.subplots(
        2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [1.25, 1]}
    )
    geometry_axis = axes[0, 0]
    color_norm = Normalize(
        vmin=float(physical_sensitivity.min()), vmax=float(physical_sensitivity.max())
    )
    points = geometry_axis.scatter(
        coords[:, 0], coords[:, 1], c=physical_sensitivity, cmap='magma', norm=color_norm,
        s=25, zorder=3,
    )
    geometry_axis.plot(coords[:, 0], coords[:, 1], color='0.45', linewidth=0.8, zorder=1)
    nonzero = physical_sensitivity > 0.0
    unit_gradients = np.zeros_like(physical_gradient_values)
    unit_gradients[nonzero] = (
        physical_gradient_values[nonzero] / physical_sensitivity[nonzero, np.newaxis]
    )
    arrow_length = max(float(np.ptp(coords[:, 0])) * 0.025, 1.0e-4)
    geometry_axis.quiver(
        coords[:, 0], coords[:, 1],
        unit_gradients[:, 0] * arrow_length, unit_gradients[:, 1] * arrow_length,
        physical_sensitivity, cmap='magma', norm=color_norm, angles='xy',
        scale_units='xy', scale=1.0, width=0.0035, zorder=4,
    )
    geometry_axis.set_aspect('equal', adjustable='box')
    geometry_axis.set_xlabel('x / chord')
    geometry_axis.set_ylabel('y / chord')
    geometry_axis.set_title('Physical geometry: arrow direction is physical input gradient')
    geometry_axis.grid(True, linestyle='--', alpha=0.3)
    figure.colorbar(points, ax=geometry_axis, label='||dD/d(x, y)|| physical')

    physical_axis = axes[0, 1]
    physical_axis.plot(point_indices, physical_gradient_values[:, 0], label='dD/dx physical', linewidth=1.3)
    physical_axis.plot(point_indices, physical_gradient_values[:, 1], label='dD/dy physical', linewidth=1.3)
    physical_axis.plot(point_indices, physical_sensitivity, label='L2 sensitivity physical', color='black', linewidth=1.4)
    physical_axis.set_xlabel('Ordered airfoil point index')
    physical_axis.set_ylabel('Derivative per physical chord coordinate')
    physical_axis.set_title('Physical-coordinate gradient components')
    physical_axis.grid(True, linestyle='--', alpha=0.3)
    physical_axis.legend()

    normalized_axis = axes[1, 0]
    normalized_axis.plot(point_indices, normalized_gradient_values[:, 0], label='dD/dx normalized', linewidth=1.3)
    normalized_axis.plot(point_indices, normalized_gradient_values[:, 1], label='dD/dy normalized', linewidth=1.3)
    normalized_axis.plot(point_indices, normalized_sensitivity, label='L2 sensitivity normalized', color='black', linewidth=1.4)
    normalized_axis.set_xlabel('Ordered airfoil point index')
    normalized_axis.set_ylabel('Derivative per GAN normalized coordinate')
    normalized_axis.set_title('Exact discriminator-input gradient components')
    normalized_axis.grid(True, linestyle='--', alpha=0.3)
    normalized_axis.legend()

    sensitivity_axis = axes[1, 1]
    sensitivity_axis.plot(coords[:, 0], physical_sensitivity, color='tab:red', linewidth=1.5)
    sensitivity_axis.scatter(coords[:, 0], physical_sensitivity, c=point_indices, cmap='viridis', s=18)
    sensitivity_axis.set_xlabel('x / chord')
    sensitivity_axis.set_ylabel('||dD/d(x, y)|| physical')
    sensitivity_axis.set_title('Sensitivity along the ordered airfoil geometry')
    sensitivity_axis.grid(True, linestyle='--', alpha=0.3)

    identifier = f'dataset index {dataset_index}' if source == 'real' else f'generated sample {sample_index}'
    figure.suptitle(f'Critic sensitivity: {identifier}, D = {float(score):.5f}')
    figure.tight_layout()
    save_report_figure(figure, path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def write_mean_csv(path, source, physical_coords, normalized_gradients,
                   physical_gradients, scores):
    mean_coords = physical_coords.mean(dim=0)
    mean_normalized_gradients = normalized_gradients.mean(dim=0)
    mean_physical_gradients = physical_gradients.mean(dim=0)
    normalized_sensitivities = torch.linalg.vector_norm(normalized_gradients, dim=2)
    physical_sensitivities = torch.linalg.vector_norm(physical_gradients, dim=2)
    surface_names = split_surface_names(mean_coords)

    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'source', 'point_index', 'surface', 'x_physical_mean', 'y_physical_mean',
                'y_physical_std',
                'mean_d_score_d_x_normalized', 'mean_d_score_d_y_normalized',
                'mean_sensitivity_normalized_l2', 'std_sensitivity_normalized_l2',
                'mean_d_score_d_x_physical', 'mean_d_score_d_y_physical',
                'norm_of_mean_physical_gradient_l2',
                'mean_sensitivity_physical_l2', 'std_sensitivity_physical_l2',
                'mean_critic_score', 'std_critic_score',
            ],
        )
        writer.writeheader()
        for point_index in range(mean_coords.shape[0]):
            writer.writerow({
                'source': source,
                'point_index': point_index,
                'surface': surface_names[point_index],
                'x_physical_mean': float(mean_coords[point_index, 0]),
                'y_physical_mean': float(mean_coords[point_index, 1]),
                'y_physical_std': float(physical_coords[:, point_index, 1].std(unbiased=False)),
                'mean_d_score_d_x_normalized': float(mean_normalized_gradients[point_index, 0]),
                'mean_d_score_d_y_normalized': float(mean_normalized_gradients[point_index, 1]),
                'mean_sensitivity_normalized_l2': float(normalized_sensitivities[:, point_index].mean()),
                'std_sensitivity_normalized_l2': float(normalized_sensitivities[:, point_index].std(unbiased=False)),
                'mean_d_score_d_x_physical': float(mean_physical_gradients[point_index, 0]),
                'mean_d_score_d_y_physical': float(mean_physical_gradients[point_index, 1]),
                'norm_of_mean_physical_gradient_l2': float(torch.linalg.vector_norm(mean_physical_gradients[point_index])),
                'mean_sensitivity_physical_l2': float(physical_sensitivities[:, point_index].mean()),
                'std_sensitivity_physical_l2': float(physical_sensitivities[:, point_index].std(unbiased=False)),
                'mean_critic_score': float(scores.mean()),
                'std_critic_score': float(scores.std(unbiased=False)),
            })


def plot_mean_sample(path, source, physical_coords, physical_gradients, scores):
    mean_coords = physical_coords.mean(dim=0).cpu().numpy()
    mean_gradients = physical_gradients.mean(dim=0).cpu().numpy()
    sensitivities = torch.linalg.vector_norm(physical_gradients, dim=2)
    mean_sensitivity = sensitivities.mean(dim=0).cpu().numpy()
    std_sensitivity = sensitivities.std(dim=0, unbiased=False).cpu().numpy()
    norm_of_mean_gradient = np.linalg.vector_norm(mean_gradients, axis=1)
    point_indices = np.arange(mean_coords.shape[0])

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    geometry_axis = axes[0]
    color_norm = Normalize(
        vmin=float(mean_sensitivity.min()), vmax=float(mean_sensitivity.max())
    )
    points = geometry_axis.scatter(
        mean_coords[:, 0], mean_coords[:, 1], c=mean_sensitivity, cmap='magma',
        norm=color_norm, s=28, zorder=3,
    )
    geometry_axis.plot(mean_coords[:, 0], mean_coords[:, 1], color='0.45', linewidth=0.8, zorder=1)
    nonzero = norm_of_mean_gradient > 0.0
    unit_gradients = np.zeros_like(mean_gradients)
    unit_gradients[nonzero] = (
        mean_gradients[nonzero] / norm_of_mean_gradient[nonzero, np.newaxis]
    )
    arrow_length = max(float(np.ptp(mean_coords[:, 0])) * 0.025, 1.0e-4)
    geometry_axis.quiver(
        mean_coords[:, 0], mean_coords[:, 1],
        unit_gradients[:, 0] * arrow_length, unit_gradients[:, 1] * arrow_length,
        mean_sensitivity, cmap='magma', norm=color_norm, angles='xy',
        scale_units='xy', scale=1.0, width=0.0035, zorder=4,
    )
    geometry_axis.set_aspect('equal', adjustable='box')
    geometry_axis.set_xlabel('mean x / chord')
    geometry_axis.set_ylabel('mean y / chord')
    geometry_axis.set_title('Mean geometry and mean-gradient direction')
    geometry_axis.grid(True, linestyle='--', alpha=0.3)
    figure.colorbar(points, ax=geometry_axis, label='mean ||dD/d(x, y)|| physical')

    curve_axis = axes[1]
    curve_axis.plot(point_indices, mean_gradients[:, 0], label='mean dD/dx physical')
    curve_axis.plot(point_indices, mean_gradients[:, 1], label='mean dD/dy physical')
    curve_axis.plot(point_indices, mean_sensitivity, color='black', label='mean L2 sensitivity')
    curve_axis.fill_between(
        point_indices,
        np.maximum(mean_sensitivity - std_sensitivity, 0.0),
        mean_sensitivity + std_sensitivity,
        color='black', alpha=0.15, label='L2 sensitivity +/- 1 std',
    )
    curve_axis.set_xlabel('Ordered airfoil point index')
    curve_axis.set_ylabel('Derivative per physical chord coordinate')
    curve_axis.set_title('Equal-weight average over selected airfoils')
    curve_axis.grid(True, linestyle='--', alpha=0.3)
    curve_axis.legend()

    figure.suptitle(
        f'Average critic sensitivity: {source}, n = {physical_coords.shape[0]}, '
        f'mean D = {float(scores.mean()):.5f}'
    )
    figure.tight_layout()
    save_report_figure(figure, path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def run_visualization(args):
    config = load_config(args.config)
    device = resolve_device(args.device)
    condition_stats = torch.load(CONDITION_NORM_PATH, map_location=device, weights_only=True)
    coord_stats = torch.load(COORD_NORM_PATH, map_location=device, weights_only=True)
    discriminator, generator = load_models(
        config, args.checkpoint, device, include_generator=args.source == 'generated'
    )

    if args.source == 'real':
        physical_coords, physical_conditions, record_indices = load_real_development_samples(
            config, args.sample_count, args.seed, device
        )
        normalized_coords = normalize_gan_coords(physical_coords, coord_stats).view(
            args.sample_count, config['num_output_points'], 2
        )
        normalized_conditions = (
            physical_conditions - condition_stats['mean']
        ) / condition_stats['std']
    else:
        normalized_coords, physical_conditions, record_indices = load_generated_samples(
            config, generator, args.sample_count, args.labels, args.seed, condition_stats, device
        )
        x_range = coord_stats['x_max'] - coord_stats['x_min']
        y_range = coord_stats['y_max'] - coord_stats['y_min']
        physical_coords = normalized_coords.clone()
        physical_coords[:, :, 0] = physical_coords[:, :, 0] * x_range + coord_stats['x_min']
        physical_coords[:, :, 1] = physical_coords[:, :, 1] * y_range + coord_stats['y_min']
        normalized_conditions = (
            physical_conditions - condition_stats['mean']
        ) / condition_stats['std']

    scores, normalized_gradients = calculate_input_gradients(
        discriminator, normalized_coords, normalized_conditions
    )
    coordinate_ranges = torch.stack(
        [coord_stats['x_max'] - coord_stats['x_min'], coord_stats['y_max'] - coord_stats['y_min']]
    ).to(device)
    if bool(torch.any(coordinate_ranges <= 0).item()):
        raise ValueError(f'Coordinate normalization ranges must be positive, got {coordinate_ranges.tolist()}')
    physical_gradients = normalized_gradients / coordinate_ranges.view(1, 1, 2)

    os.makedirs(args.output_directory, exist_ok=True)
    report_samples = []
    for sample_index in range(args.sample_count):
        stem = f'{args.source}_{sample_index:03d}'
        csv_path = os.path.join(args.output_directory, f'{stem}.csv')
        plot_path = os.path.join(args.output_directory, f'{stem}.png')
        write_sample_csv(
            csv_path, args.source, sample_index, record_indices[sample_index],
            physical_coords[sample_index].cpu(), normalized_gradients[sample_index].cpu(),
            physical_gradients[sample_index].cpu(), scores[sample_index, 0].cpu(),
        )
        plot_sample(
            plot_path, args.source, sample_index, record_indices[sample_index],
            physical_coords[sample_index].cpu(), normalized_gradients[sample_index].cpu(),
            physical_gradients[sample_index].cpu(), scores[sample_index, 0].cpu(),
        )
        report_samples.append({
            'sample_index': sample_index,
            'dataset_index': record_indices[sample_index] if args.source == 'real' else None,
            'critic_score': float(scores[sample_index, 0].cpu()),
            'max_physical_sensitivity': float(torch.linalg.vector_norm(physical_gradients[sample_index], dim=1).max().cpu()),
            'csv_path': csv_path,
            'plot_path': plot_path,
        })

    mean_csv_path = os.path.join(args.output_directory, 'mean.csv')
    mean_plot_path = os.path.join(args.output_directory, 'mean.png')
    write_mean_csv(
        mean_csv_path, args.source, physical_coords.cpu(), normalized_gradients.cpu(),
        physical_gradients.cpu(), scores[:, 0].cpu(),
    )
    plot_mean_sample(
        mean_plot_path, args.source, physical_coords.cpu(), physical_gradients.cpu(),
        scores[:, 0].cpu(),
    )

    save_yaml(os.path.join(args.output_directory, 'report.yaml'), {
        'source': args.source,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'sample_count': args.sample_count,
        'seed': args.seed,
        'gradient_definition': 'd D(coords, normalized_condition) / d coords',
        'coordinate_ranges': {
            'x': float(coordinate_ranges[0].cpu()),
            'y': float(coordinate_ranges[1].cpu()),
        },
        'mean_csv_path': mean_csv_path,
        'mean_plot_path': mean_plot_path,
        'samples': report_samples,
    })
    print(f'Wrote discriminator sensitivity report to {args.output_directory}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize per-point critic coordinate gradients using autograd'
    )
    parser.add_argument('--config', default='config.yaml', help='YAML configuration path')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_PATH, help='GAN checkpoint path')
    parser.add_argument('--output-directory', default=DEFAULT_OUTPUT_DIRECTORY, help='Report directory')
    parser.add_argument('--source', choices=('real', 'generated'), default='real')
    parser.add_argument('--sample-count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260704)
    parser.add_argument('--labels', type=float, nargs='+', help='Generated source condition: alpha Re CL CM')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    run_visualization(parser.parse_args())


if __name__ == '__main__':
    main()
