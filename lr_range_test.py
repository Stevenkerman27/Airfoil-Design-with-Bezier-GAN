import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from model import AerodynamicSurrogate
from surrogate_split import load_cross_validation_manifest, resolve_surrogate_dataset_config
from train_surrogate import (
    AirfoilSurrogateDataset,
    build_weighted_mse_loss,
    compute_global_gradient_norm,
    load_config,
    resolve_device,
)


LR_RANGE_PLOT_PATH = 'model/surrogate_lr_range.png'
LR_RANGE_CONFIG_KEYS = [
    'surrogate_lr_range_start',
    'surrogate_lr_range_end',
    'surrogate_lr_range_epochs',
    'surrogate_lr_range_runs',
    'surrogate_lr_range_ema_beta',
    'surrogate_lr_range_max_loss_multiplier',
]


def validate_lr_range_config(config):
    missing_keys = [key for key in LR_RANGE_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f'LR range test configuration is missing keys: {missing_keys}')
    start_lr = float(config['surrogate_lr_range_start'])
    end_lr = float(config['surrogate_lr_range_end'])
    epochs = config['surrogate_lr_range_epochs']
    runs = config['surrogate_lr_range_runs']
    ema_beta = float(config['surrogate_lr_range_ema_beta'])
    max_loss_multiplier = float(config['surrogate_lr_range_max_loss_multiplier'])
    if start_lr <= 0.0 or end_lr <= start_lr:
        raise ValueError(f'LR range bounds must satisfy 0 < start < end, got {start_lr}, {end_lr}')
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f'surrogate_lr_range_epochs must be a positive integer, got {epochs}')
    if not isinstance(runs, int) or runs <= 0:
        raise ValueError(f'surrogate_lr_range_runs must be a positive integer, got {runs}')
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError(f'surrogate_lr_range_ema_beta must be in [0, 1), got {ema_beta}')
    if max_loss_multiplier <= 1.0:
        raise ValueError(
            'surrogate_lr_range_max_loss_multiplier must be greater than 1, '
            f'got {max_loss_multiplier}'
        )


def exponential_learning_rate(start_lr, end_lr, step, total_steps):
    if total_steps <= 1:
        raise ValueError(f'total_steps must exceed 1, got {total_steps}')
    if not 0 <= step < total_steps:
        raise ValueError(f'step must be in [0, {total_steps}), got {step}')
    progress = step / (total_steps - 1)
    return start_lr * ((end_lr / start_lr) ** progress)


def update_ema(previous_ema, value, beta, step):
    uncorrected_ema = beta * previous_ema + (1.0 - beta) * value
    return uncorrected_ema, uncorrected_ema / (1.0 - (beta ** (step + 1)))


def set_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = learning_rate


def aggregate_smoothed_losses(records_by_run):
    maximum_steps = max(len(records) for records in records_by_run)
    learning_rates = []
    mean_smoothed_losses = []
    for step in range(maximum_steps):
        step_records = [records[step] for records in records_by_run if len(records) > step]
        learning_rates.append(step_records[0]['learning_rate'])
        mean_smoothed_losses.append(
            sum(record['smoothed_loss'] for record in step_records) / len(step_records)
        )
    return learning_rates, mean_smoothed_losses


def plot_lr_range(records_by_run, path):
    plt.figure(figsize=(10, 6))
    for run_index, records in enumerate(records_by_run, start=1):
        learning_rates = [record['learning_rate'] for record in records]
        smoothed_losses = [record['smoothed_loss'] for record in records]
        plt.plot(learning_rates, smoothed_losses, alpha=0.45, linewidth=1, label=f'Run {run_index} EMA')
    learning_rates, mean_smoothed_losses = aggregate_smoothed_losses(records_by_run)
    plt.plot(learning_rates, mean_smoothed_losses, color='black', linewidth=2.5, label='Mean EMA')
    plt.xlabel('Learning rate')
    plt.ylabel('Weighted MSE loss')
    plt.title('Surrogate Learning Rate Range Test')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def run_single_lr_range_test(config, train_loader, device, total_steps):
    model = AerodynamicSurrogate(config).to(device)
    criterion = build_weighted_mse_loss(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['surrogate_lr_range_start']),
        weight_decay=float(config['surrogate_weight_decay']),
    )
    start_lr = float(config['surrogate_lr_range_start'])
    end_lr = float(config['surrogate_lr_range_end'])
    ema_beta = float(config['surrogate_lr_range_ema_beta'])
    max_loss_multiplier = float(config['surrogate_lr_range_max_loss_multiplier'])
    records = []
    previous_ema = 0.0
    best_smoothed_loss = math.inf
    stop_reason = 'completed'
    step = 0
    model.train()

    for _ in range(config['surrogate_lr_range_epochs']):
        for coords, conditions, targets in train_loader:
            learning_rate = exponential_learning_rate(start_lr, end_lr, step, total_steps)
            set_learning_rate(optimizer, learning_rate)
            coords = coords.to(device)
            conditions = conditions.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(coords, conditions)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                stop_reason = 'non_finite_loss'
                break
            loss.backward()
            gradient_norm = compute_global_gradient_norm(model.parameters())
            if not math.isfinite(gradient_norm):
                stop_reason = 'non_finite_gradient'
                break
            optimizer.step()

            previous_ema, smoothed_loss = update_ema(previous_ema, loss.item(), ema_beta, step)
            record = {
                'step': step + 1,
                'learning_rate': learning_rate,
                'loss': loss.item(),
                'smoothed_loss': smoothed_loss,
                'gradient_norm': gradient_norm,
            }
            records.append(record)
            best_smoothed_loss = min(best_smoothed_loss, smoothed_loss)
            if smoothed_loss > max_loss_multiplier * best_smoothed_loss:
                stop_reason = 'loss_diverged'
                break
            step += 1
        if stop_reason != 'completed':
            break

    if not records:
        raise ValueError(f'LR range test ended without finite measurements: {stop_reason}')
    return records, stop_reason


def run_lr_range_test(config_path):
    config = load_config(config_path)
    validate_lr_range_config(config)
    device = resolve_device(config)
    dataset_config = resolve_surrogate_dataset_config(config)
    raw_data = torch.load(dataset_config['data_path'], weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    dataset, _ = AirfoilSurrogateDataset.from_training_indices(
        raw_data,
        manifest['development_indices'],
    )
    train_set = Subset(dataset, manifest['development_indices'])
    train_loader = DataLoader(
        train_set,
        batch_size=config['surrogate_batch_size'],
        shuffle=True,
        drop_last=False,
    )
    total_steps = config['surrogate_lr_range_epochs'] * len(train_loader)
    if total_steps <= 1:
        raise ValueError(f'LR range test requires at least two batches, got {total_steps}')

    records_by_run = []
    results = []
    for run_index in range(config['surrogate_lr_range_runs']):
        records, stop_reason = run_single_lr_range_test(config, train_loader, device, total_steps)
        records_by_run.append(records)
        best_record = min(records, key=lambda record: record['smoothed_loss'])
        results.append({
            'run': run_index + 1,
            'completed_steps': len(records),
            'stop_reason': stop_reason,
            'minimum_smoothed_loss': best_record['smoothed_loss'],
            'minimum_smoothed_loss_lr': best_record['learning_rate'],
        })
    plot_lr_range(records_by_run, LR_RANGE_PLOT_PATH)
    result = {
        'dataset_name': 'airfoil_group_development',
        'scheduled_steps': total_steps,
        'runs': results,
        'plot_path': LR_RANGE_PLOT_PATH,
    }
    print('LR range test dataset: airfoil_group_development')
    for run_result in results:
        print(
            f"Run {run_result['run']}: steps={run_result['completed_steps']}/{total_steps}, "
            f"stop={run_result['stop_reason']}, "
            f"min EMA={run_result['minimum_smoothed_loss']:.6f} "
            f"at lr={run_result['minimum_smoothed_loss_lr']:.8g}"
        )
    print(f"Saved plot to {LR_RANGE_PLOT_PATH}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an LR range test for the aerodynamic surrogate')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    args = parser.parse_args()
    run_lr_range_test(args.config)
