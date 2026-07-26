import argparse
import concurrent.futures
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from artifact_io import ensure_parent_directory, save_report_figure, save_yaml
from foildata.xfoil import run_xfoil_single
from gan_exploration import (
    SURROGATE_TARGET_ORDER,
    batch_random_noise,
    generate_and_predict,
    load_development_condition_data,
    load_exploration_models,
    optimize_latent_noise,
    resolve_exploration_config,
    sample_extended_conditions,
)
from train_surrogate import load_config, resolve_device, set_training_seed
from utils import normalize_airfoil_chord_coordinates


CONDITION_REPORT_PATH = 'reports/exploration/gan_condition_statistics.yaml'
NOISE_REPORT_PATH = 'reports/exploration/gan_noise_xfoil_report.yaml'
NOISE_RECORDS_PATH = 'model/gan_noise_xfoil_records.pt'
NOISE_PLOT_PATH = 'reports/exploration/gan_noise_xfoil_spread.png'


def mode_output_path(path, mode):
    if mode == 'random':
        return path
    root, extension = os.path.splitext(path)
    return f'{root}_{mode}{extension}'


def evaluate_xfoil_record(task):
    record, timeout_seconds = task
    try:
        coords = normalize_airfoil_chord_coordinates(record['coords']).numpy()
        result = run_xfoil_single(
            coords,
            float(record['condition'][1].item()),
            float(record['condition'][0].item()),
            timeout=timeout_seconds,
            return_all=True,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        record['xfoil_status'] = 'preprocess_failed'
        record['failure_reason'] = str(error)
        return record
    if result is None:
        record['xfoil_status'] = 'xfoil_failed'
        return record
    targets = torch.tensor(
        [result['CM'], result['CL'], result['CD']], dtype=torch.float32
    )
    if not bool(torch.isfinite(targets).all().item()):
        record['xfoil_status'] = 'xfoil_failed'
        record['failure_reason'] = 'non_finite_xfoil_result'
        return record
    record['coords'] = torch.from_numpy(coords).float()
    record['xfoil_status'] = 'success'
    record['xfoil_targets'] = targets
    return record


def _metric_summary(values):
    if not values:
        return None
    tensor = torch.stack(values).float()
    return {
        target_name: {
            'mean': float(tensor[:, index].mean().item()),
            'std': float(tensor[:, index].std(unbiased=False).item()),
            'min': float(tensor[:, index].min().item()),
            'max': float(tensor[:, index].max().item()),
        }
        for index, target_name in enumerate(SURROGATE_TARGET_ORDER)
    }


def build_noise_report(records, sampled_conditions, mode):
    by_condition = {index: [] for index in range(len(sampled_conditions))}
    for record in records:
        by_condition[record['condition_index']].append(record)
    condition_reports = []
    successful_targets = []
    surrogate_errors = []
    for condition_index, condition_records in by_condition.items():
        success_records = [
            record for record in condition_records if record['xfoil_status'] == 'success'
        ]
        xfoil_targets = [record['xfoil_targets'] for record in success_records]
        successful_targets.extend(xfoil_targets)
        surrogate_errors.extend([
            torch.abs(record['surrogate_prediction'] - record['xfoil_targets'])
            for record in success_records
        ])
        condition = sampled_conditions[condition_index]
        condition_reports.append({
            'condition_index': condition_index,
            'condition_source': condition['source'],
            'source_dataset_index': condition['source_dataset_index'],
            'mahalanobis_radius': condition['mahalanobis_radius'],
            'alpha': float(condition['condition'][0].item()),
            'Re': float(condition['condition'][1].item()),
            'target_CL': float(condition['condition'][2].item()),
            'target_CM': float(condition['condition'][3].item()),
            'xfoil_success_count': len(success_records),
            'xfoil_attempt_count': len(condition_records),
            'xfoil_aerodynamic_spread': _metric_summary(xfoil_targets),
        })
    status_counts = {}
    for record in records:
        status = record['xfoil_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        'mode': mode,
        'target_order': SURROGATE_TARGET_ORDER,
        'requested_sample_count': len(records),
        'xfoil_successful_count': len(successful_targets),
        'xfoil_convergence_rate': len(successful_targets) / len(records),
        'status_counts': status_counts,
        'overall_xfoil_aerodynamic_spread': _metric_summary(successful_targets),
        'overall_surrogate_mae_to_xfoil': _metric_summary(surrogate_errors),
        'conditions': condition_reports,
    }
    return report


def plot_noise_spread(report, path):
    target_names = SURROGATE_TARGET_ORDER
    figure, axes = plt.subplots(len(target_names), 1, figsize=(11, 9), sharex=True)
    for axis, target_name in zip(axes, target_names):
        positions = []
        spreads = []
        colors = []
        for condition in report['conditions']:
            spread = condition['xfoil_aerodynamic_spread']
            if spread is None:
                continue
            positions.append(condition['condition_index'])
            spreads.append(spread[target_name]['std'])
            colors.append(
                '#2678b2' if condition['condition_source'] == 'empirical' else '#d66b28'
            )
        axis.scatter(positions, spreads, c=colors, s=36)
        axis.set_ylabel(f'XFoil std {target_name}')
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Expanded condition index')
    figure.tight_layout()
    ensure_parent_directory(path)
    save_report_figure(figure, path, dpi=250)
    plt.close(figure)


def generate_random_records(
    generator,
    surrogate,
    auxiliary_stats,
    config,
    sampled_conditions,
):
    exploration_config = resolve_exploration_config(config)
    samples_per_condition = exploration_config['noise_samples_per_condition']
    conditions = torch.stack([
        condition['condition']
        for condition in sampled_conditions
        for _ in range(samples_per_condition)
    ])
    noise = batch_random_noise(
        len(sampled_conditions),
        samples_per_condition,
        config['noise_dimension'],
        config['surrogate_seed'],
    )
    coords_batches = []
    prediction_batches = []
    device = next(generator.parameters()).device
    batch_size = exploration_config['generation_batch_size']
    with torch.no_grad():
        for start in range(0, len(conditions), batch_size):
            coords, predictions = generate_and_predict(
                generator,
                surrogate,
                auxiliary_stats,
                config,
                noise[start:start + batch_size].to(device),
                conditions[start:start + batch_size].to(device),
            )
            coords_batches.append(coords.cpu())
            prediction_batches.append(predictions.cpu())
    coords = torch.cat(coords_batches, dim=0)
    predictions = torch.cat(prediction_batches, dim=0)
    records = []
    for index in range(len(conditions)):
        records.append({
            'condition_index': index // samples_per_condition,
            'noise_index': index % samples_per_condition,
            'noise_mode': 'random',
            'condition': conditions[index].clone(),
            'noise': noise[index].clone(),
            'coords': coords[index].clone(),
            'surrogate_prediction': predictions[index].clone(),
        })
    return records


def generate_optimized_records(
    generator,
    surrogate,
    auxiliary_stats,
    config,
    sampled_conditions,
):
    exploration_config = resolve_exploration_config(config)
    records = []
    for condition_index, sampled_condition in enumerate(sampled_conditions):
        result = optimize_latent_noise(
            generator,
            surrogate,
            auxiliary_stats,
            config,
            sampled_condition['condition'],
            config['surrogate_seed'] + condition_index,
        )
        for noise_index in range(result['optimized_noise'].size(0)):
            records.append({
                'condition_index': condition_index,
                'noise_index': noise_index,
                'noise_mode': 'optimized',
                'condition': result['condition'].clone(),
                'noise': result['optimized_noise'][noise_index].clone(),
                'initial_noise': result['initial_noise'][noise_index].clone(),
                'coords': result['coords'][noise_index].clone(),
                'surrogate_prediction': result['surrogate_predictions'][noise_index].clone(),
            })
    return records


def run_exploration(config_path, mode):
    config = load_config(config_path)
    exploration_config = resolve_exploration_config(config)
    if not isinstance(config['max_workers'], int) or config['max_workers'] <= 0:
        raise ValueError('max_workers must be a positive integer')
    set_training_seed(config['surrogate_seed'])
    raw_data, development_indices = load_development_condition_data(config)
    sampled_conditions, condition_statistics = sample_extended_conditions(
        raw_data,
        development_indices,
        exploration_config['condition_count'],
        float(exploration_config['empirical_condition_fraction']),
        float(exploration_config['mahalanobis_radius']),
        config['surrogate_seed'],
    )
    condition_statistics['sampling'] = {
        'condition_count': len(sampled_conditions),
        'empirical_condition_fraction': float(
            exploration_config['empirical_condition_fraction']
        ),
        'mahalanobis_radius_limit': float(exploration_config['mahalanobis_radius']),
        'sampled_conditions': [
            {
                'source': sample['source'],
                'source_dataset_index': sample['source_dataset_index'],
                'mahalanobis_radius': sample['mahalanobis_radius'],
                'alpha': float(sample['condition'][0].item()),
                'Re': float(sample['condition'][1].item()),
                'CL': float(sample['condition'][2].item()),
                'CM': float(sample['condition'][3].item()),
            }
            for sample in sampled_conditions
        ],
    }
    save_yaml(CONDITION_REPORT_PATH, condition_statistics)

    device = resolve_device(config)
    generator, surrogate, auxiliary_stats = load_exploration_models(config, device)
    if mode == 'random':
        records = generate_random_records(
            generator, surrogate, auxiliary_stats, config, sampled_conditions
        )
    elif mode == 'optimized':
        records = generate_optimized_records(
            generator, surrogate, auxiliary_stats, config, sampled_conditions
        )
    else:
        raise ValueError(f'Unsupported exploration mode: {mode}')
    tasks = [
        (record, exploration_config['xfoil_timeout_seconds']) for record in records
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config['max_workers']
    ) as executor:
        records = list(executor.map(evaluate_xfoil_record, tasks))
    report = build_noise_report(records, sampled_conditions, mode)
    report_path = mode_output_path(NOISE_REPORT_PATH, mode)
    record_path = mode_output_path(NOISE_RECORDS_PATH, mode)
    plot_path = mode_output_path(NOISE_PLOT_PATH, mode)
    save_yaml(report_path, report)
    ensure_parent_directory(record_path)
    torch.save(records, record_path)
    plot_noise_spread(report, plot_path)
    print(f'Condition statistics: {CONDITION_REPORT_PATH}')
    print(f'XFoil noise report: {report_path}')
    print(
        f'XFoil convergence: {report["xfoil_successful_count"]}/'
        f'{report["requested_sample_count"]} '
        f'({report["xfoil_convergence_rate"]:.2%})'
    )
    return condition_statistics, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Explore GAN condition statistics and latent-noise aerodynamics with XFoil'
    )
    parser.add_argument('--config', default='config.yaml', help='Path to config yaml')
    parser.add_argument(
        '--mode', choices=['random', 'optimized'], default='random',
        help='Random noise sensitivity or surrogate-guided low-drag candidate audit',
    )
    arguments = parser.parse_args()
    run_exploration(arguments.config, arguments.mode)
