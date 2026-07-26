import argparse
import torch

from artifact_io import save_yaml
from model import AerodynamicSurrogate
from surrogate_split import load_cross_validation_manifest
from train_surrogate import (
    AirfoilSurrogateDataset,
    SURROGATE_TARGET_NAMES,
    build_weighted_mse_loss,
    evaluate,
    load_config,
    plot_prediction_scatter,
    resolve_device,
)


EVAL_PLOT_PATHS = {
    'CL': 'reports/surrogate/surrogate_test_cl.png',
    'CD': 'reports/surrogate/surrogate_test_cd.png',
    'CM': 'reports/surrogate/surrogate_test_cm.png',
}
SURROGATE_DATASET_PATH = 'model/airfoil_dataset.pt'
SURROGATE_NORM_PATH = 'model/surrogate_airfoil_group_norm.pt'
SURROGATE_BEST_MODEL_PATH = 'model/surrogate_airfoil_group_best.pt'


def load_surrogate_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    supported_selection_policies = {
        'fixed_final_epoch',
        'generated_validation_weighted_mse',
        'alternating_fixed_epoch',
    }
    if checkpoint['selection_policy'] not in supported_selection_policies:
        raise ValueError(
            f"Unexpected surrogate checkpoint selection policy: {checkpoint['selection_policy']}"
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def compute_target_metrics(predictions, targets, target_names):
    errors = predictions - targets
    per_target_mae = torch.mean(torch.abs(errors), dim=0)
    per_target_rmse = torch.sqrt(torch.mean(errors ** 2, dim=0))
    target_mean = torch.mean(targets, dim=0)
    ss_res = torch.sum(errors ** 2, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    if torch.any(ss_tot <= 0):
        raise ValueError('R2 is undefined when a target has zero variance')
    return {
        'target_order': target_names,
        'per_target_mae': {
            name: float(per_target_mae[index].item()) for index, name in enumerate(target_names)
        },
        'per_target_rmse': {
            name: float(per_target_rmse[index].item()) for index, name in enumerate(target_names)
        },
        'per_target_r2': {
            name: float((1.0 - ss_res[index] / ss_tot[index]).item())
            for index, name in enumerate(target_names)
        },
    }


def save_metrics(path, metrics):
    save_yaml(path, metrics)


def run_evaluation(config_path, model_path=None, metrics_path=None):
    config = load_config(config_path)
    device = resolve_device(config)
    raw_data = torch.load(SURROGATE_DATASET_PATH, weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    dataset = AirfoilSurrogateDataset.from_norm_path(
        raw_data, SURROGATE_NORM_PATH, device
    )
    test_indices = dataset.prepare_indices(manifest['test_indices'])
    checkpoint_path = model_path if model_path is not None else SURROGATE_BEST_MODEL_PATH
    model = AerodynamicSurrogate(config).to(device)
    checkpoint = load_surrogate_checkpoint(model, checkpoint_path, device)
    result = evaluate(
        model,
        dataset,
        test_indices,
        build_weighted_mse_loss(config, device),
        config['surrogate_batch_size'],
        device,
    )
    metrics = {
        'split': 'test',
        'sample_count': len(manifest['test_indices']),
        'model_path': checkpoint_path,
        'training_epoch_count': int(checkpoint['training_epoch_count']),
        'selection_policy': checkpoint['selection_policy'],
        'eval_loss': float(result['loss']),
        'eval_mae': float(result['mae']),
    }
    metrics.update(compute_target_metrics(result['predictions'], result['targets'], SURROGATE_TARGET_NAMES))
    for target_name, path in EVAL_PLOT_PATHS.items():
        plot_prediction_scatter(
            result['targets'],
            result['predictions'],
            SURROGATE_TARGET_NAMES.index(target_name),
            target_name,
            path,
        )
    output_path = metrics_path or 'reports/surrogate/surrogate_test_metrics.yaml'
    save_metrics(output_path, metrics)
    print(f"Independent test sample count: {metrics['sample_count']}")
    print(f"Independent test loss: {metrics['eval_loss']:.6f}")
    for target_name in SURROGATE_TARGET_NAMES:
        print(
            f"{target_name} MAE: {metrics['per_target_mae'][target_name]:.6f}, "
            f"RMSE: {metrics['per_target_rmse'][target_name]:.6f}, "
            f"R2: {metrics['per_target_r2'][target_name]:.6f}"
        )
    print(f'Saved metrics to {output_path}')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate final surrogate on its independent test set')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--model', type=str, help='Override surrogate checkpoint path')
    parser.add_argument('--metrics', type=str, help='Override metrics yaml output path')
    args = parser.parse_args()
    run_evaluation(args.config, model_path=args.model, metrics_path=args.metrics)
