import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from model import AerodynamicSurrogate
from train_surrogate import (
    AirfoilSurrogateDataset,
    SURROGATE_TARGET_NAMES,
    build_weighted_mse_loss,
    evaluate,
    load_config,
    plot_prediction_scatter,
    resolve_device,
    split_dataset,
)


EVAL_PLOT_PATHS = {
    'train': {
        'CL': 'model/surrogate_train_cl.png',
        'CD': 'model/surrogate_train_cd.png',
        'CM': 'model/surrogate_train_cm.png',
    },
    'val': {
        'CL': 'model/surrogate_eval_val_cl.png',
        'CD': 'model/surrogate_eval_val_cd.png',
        'CM': 'model/surrogate_eval_val_cm.png',
    },
    'test': {
        'CL': 'model/surrogate_test_cl.png',
        'CD': 'model/surrogate_test_cd.png',
        'CM': 'model/surrogate_test_cm.png',
    },
    'all': {
        'CL': 'model/surrogate_all_cl.png',
        'CD': 'model/surrogate_all_cd.png',
        'CM': 'model/surrogate_all_cm.png',
    },
}


def load_surrogate_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def select_eval_dataset(dataset, config, split_name):
    if split_name == 'all':
        return dataset
    train_set, val_set, test_set = split_dataset(dataset, config)
    split_map = {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }
    return split_map[split_name]


def compute_target_metrics(predictions, targets, target_names):
    errors = predictions - targets
    abs_errors = torch.abs(errors)
    squared_errors = errors ** 2

    per_target_mae = torch.mean(abs_errors, dim=0)
    per_target_rmse = torch.sqrt(torch.mean(squared_errors, dim=0))
    target_mean = torch.mean(targets, dim=0)
    ss_res = torch.sum(squared_errors, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    if torch.any(ss_tot <= 0):
        raise ValueError('R2 is undefined when a target has zero variance')
    per_target_r2 = 1.0 - ss_res / ss_tot

    metrics = {
        'target_order': target_names,
        'per_target_mae': {},
        'per_target_rmse': {},
        'per_target_r2': {},
    }
    for index, name in enumerate(target_names):
        metrics['per_target_mae'][name] = float(per_target_mae[index].item())
        metrics['per_target_rmse'][name] = float(per_target_rmse[index].item())
        metrics['per_target_r2'][name] = float(per_target_r2[index].item())
    return metrics


def save_metrics(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(metrics, f, allow_unicode=True, sort_keys=False)


def run_evaluation(config_path, split_name, model_path=None, metrics_path=None):
    config = load_config(config_path)
    device = resolve_device(config)
    checkpoint_path = model_path
    if checkpoint_path is None:
        checkpoint_path = config['surrogate_best_model_path']

    dataset = AirfoilSurrogateDataset(
        config['surrogate_dataset_path'],
        config['surrogate_norm_path'],
        config,
        save_norm=False,
    )
    eval_set = select_eval_dataset(dataset, config, split_name)
    dataloader = DataLoader(
        eval_set,
        batch_size=config['surrogate_batch_size'],
        shuffle=False,
        drop_last=False,
    )

    model = AerodynamicSurrogate(config).to(device)
    checkpoint = load_surrogate_checkpoint(model, checkpoint_path, device)
    criterion = build_weighted_mse_loss(config, device)
    result = evaluate(model, dataloader, criterion, dataset, device)

    metrics = {
        'split': split_name,
        'sample_count': len(eval_set),
        'model_path': checkpoint_path,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_val_loss': float(checkpoint['val_loss']),
        'eval_loss': float(result['loss']),
        'eval_mae': float(result['mae']),
    }
    metrics.update(
        compute_target_metrics(
            result['predictions'],
            result['targets'],
            SURROGATE_TARGET_NAMES,
        )
    )

    plot_paths = EVAL_PLOT_PATHS[split_name]
    for target_name, path in plot_paths.items():
        plot_prediction_scatter(
            result['targets'],
            result['predictions'],
            SURROGATE_TARGET_NAMES.index(target_name),
            target_name,
            path,
        )

    output_metrics_path = metrics_path
    if output_metrics_path is None:
        output_metrics_path = f"model/surrogate_eval_{split_name}_metrics.yaml"
    save_metrics(output_metrics_path, metrics)

    print(f"Evaluated split: {split_name}")
    print(f"Sample count: {len(eval_set)}")
    print(f"Evaluation loss: {result['loss']:.6f}")
    print(f"Evaluation MAE: {result['mae']:.6f}")
    for target_name in SURROGATE_TARGET_NAMES:
        mae = metrics['per_target_mae'][target_name]
        rmse = metrics['per_target_rmse'][target_name]
        r2 = metrics['per_target_r2'][target_name]
        print(f"{target_name} MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")
    print(f"Saved metrics to {output_metrics_path}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained aerodynamic surrogate model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to evaluate',
    )
    parser.add_argument('--model', type=str, help='Override surrogate checkpoint path')
    parser.add_argument('--metrics', type=str, help='Override metrics yaml output path')
    args = parser.parse_args()
    run_evaluation(args.config, args.split, model_path=args.model, metrics_path=args.metrics)
