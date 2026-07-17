import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from model import AerodynamicSurrogate
from surrogate_split import load_split_indices, resolve_surrogate_dataset_config


DATASET_LABEL_NAMES = ['alpha', 'Re', 'CL', 'CM']
DATASET_CD_KEY = 'cd'
SURROGATE_CONDITION_NAMES = ['alpha', 'Re']
SURROGATE_TARGET_NAMES = ['CM', 'CL', 'CD']
LOSS_PLOT_PATH = 'model/surrogate_loss.png'
ERROR_PLOT_PATH = 'model/surrogate_error.png'
TRAINING_METRICS_PATH = 'model/surrogate_training_metrics.csv'
CLR_CONFIG_KEYS = (
    'surrogate_clr_mode',
    'surrogate_clr_base_lr',
    'surrogate_clr_max_lr',
    'surrogate_clr_step_size_epochs',
)
CLR_MODE = 'triangular2'
VALIDATION_PLOT_PATHS = {
    'CL': 'model/surrogate_val_cl.png',
    'CD': 'model/surrogate_val_cd.png',
    'CM': 'model/surrogate_val_cm.png',
}
TRAINING_METRIC_FIELDS = (
    ['epoch', 'train_loss', 'val_loss', 'train_mae', 'val_mae']
    + [f'{split}_{target.lower()}_mse' for split in ['train', 'val'] for target in SURROGATE_TARGET_NAMES]
    + [f'{split}_{target.lower()}_mae' for split in ['train', 'val'] for target in SURROGATE_TARGET_NAMES]
    + ['learning_rate', 'train_grad_norm_mean', 'train_grad_norm_max']
)


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights.float())

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got "
                f"{predictions.shape} and {targets.shape}"
            )
        squared_error = (predictions - targets) ** 2
        return torch.mean(squared_error * self.weights)


class AirfoilSurrogateDataset(Dataset):
    def __init__(self, data_path, norm_path, config, save_norm=True):
        raw_data = torch.load(data_path, weights_only=True)
        if len(raw_data) == 0:
            raise ValueError(f"Dataset is empty: {data_path}")
        self.dataset_name, self.split_indices = load_split_indices(raw_data, config)

        self.label_names = DATASET_LABEL_NAMES
        self.cd_key = DATASET_CD_KEY
        self.condition_names = SURROGATE_CONDITION_NAMES
        self.target_names = SURROGATE_TARGET_NAMES
        self.condition_indices = [self.label_names.index(name) for name in self.condition_names]

        train_items = [raw_data[index] for index in self.split_indices['train']]
        coords_train = torch.stack([item['x'] for item in train_items]).float()
        coords_train = coords_train.view(coords_train.size(0), -1, 2)
        self.x_min = coords_train[:, :, 0].min()
        self.x_max = coords_train[:, :, 0].max()
        self.y_min = coords_train[:, :, 1].min()
        self.y_max = coords_train[:, :, 1].max()

        conditions_train = torch.stack([self.extract_conditions(item) for item in train_items]).float()
        targets_train = torch.stack([self.extract_targets(item) for item in train_items])

        self.condition_mean = conditions_train.mean(dim=0)
        self.condition_std = conditions_train.std(dim=0) + 1e-8
        self.target_mean = targets_train.mean(dim=0)
        self.target_std = targets_train.std(dim=0) + 1e-8

        if save_norm:
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            torch.save({
                'source_split': 'train',
                'dataset_name': self.dataset_name,
                'split_seed': config['surrogate_seed'],
                'split_ratio': config['surrogate_split_ratio'],
                'split_counts': {
                    'train': len(self.split_indices['train']),
                    'val': len(self.split_indices['val']),
                    'test': len(self.split_indices['test']),
                },
                'coord': {
                    'x_min': self.x_min,
                    'x_max': self.x_max,
                    'y_min': self.y_min,
                    'y_max': self.y_max,
                },
                'condition': {
                    'names': self.condition_names,
                    'mean': self.condition_mean,
                    'std': self.condition_std,
                },
                'target': {
                    'names': self.target_names,
                    'mean': self.target_mean,
                    'std': self.target_std,
                },
            }, norm_path)

        self.samples = []
        for item in raw_data:
            coords = item['x'].float().clone().view(-1, 2)
            coords[:, 0] = (coords[:, 0] - self.x_min) / (self.x_max - self.x_min + 1e-8)
            coords[:, 1] = (coords[:, 1] - self.y_min) / (self.y_max - self.y_min + 1e-8)
            norm_coords = coords.view(-1)

            condition = self.extract_conditions(item)
            norm_condition = (condition - self.condition_mean) / self.condition_std

            target = self.extract_targets(item)
            norm_target = (target - self.target_mean) / self.target_std

            self.samples.append({
                'coords': norm_coords,
                'conditions': norm_condition,
                'targets': norm_target,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['coords'], sample['conditions'], sample['targets']

    def extract_conditions(self, item):
        return item['y'][self.condition_indices].float()

    def extract_targets(self, item):
        values = []
        for name in self.target_names:
            if name == 'CD':
                values.append(item[self.cd_key])
            else:
                values.append(item['y'][self.label_names.index(name)])
        return torch.tensor(values, dtype=torch.float32)

    def denormalize_targets(self, values):
        mean = self.target_mean.to(values.device)
        std = self.target_std.to(values.device)
        return values * std + mean


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(config):
    device_cfg = config['device']
    if device_cfg.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_cfg.lower() == 'cpu':
        return torch.device('cpu')
    if device_cfg.lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(f"Unknown device configuration: {device_cfg}")


def split_dataset(dataset, config):
    return (
        Subset(dataset, dataset.split_indices['train']),
        Subset(dataset, dataset.split_indices['val']),
        Subset(dataset, dataset.split_indices['test']),
    )


def evaluate(model, dataloader, criterion, dataset, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    batch_count = 0
    sample_count = 0
    target_squared_error_sum = torch.zeros(len(SURROGATE_TARGET_NAMES), device=device)
    target_absolute_error_sum = torch.zeros(len(SURROGATE_TARGET_NAMES), device=device)
    predictions = []
    targets = []

    with torch.no_grad():
        for coords, conditions, target in dataloader:
            coords = coords.to(device)
            conditions = conditions.to(device)
            target = target.to(device)

            pred = model(coords, conditions)
            loss = criterion(pred, target)

            pred_real = dataset.denormalize_targets(pred)
            target_real = dataset.denormalize_targets(target)
            mae = torch.mean(torch.abs(pred_real - target_real))

            total_loss += loss.item()
            total_mae += mae.item()
            batch_count += 1
            sample_count += target.size(0)
            target_squared_error_sum += torch.sum((pred - target) ** 2, dim=0)
            target_absolute_error_sum += torch.sum(torch.abs(pred_real - target_real), dim=0)
            predictions.append(pred_real.cpu())
            targets.append(target_real.cpu())

    if batch_count == 0:
        raise ValueError('Evaluation dataloader produced zero batches')

    return {
        'loss': total_loss / batch_count,
        'mae': total_mae / batch_count,
        'per_target_mse': target_squared_error_sum.div(sample_count).cpu(),
        'per_target_mae': target_absolute_error_sum.div(sample_count).cpu(),
        'predictions': torch.cat(predictions, dim=0),
        'targets': torch.cat(targets, dim=0),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, lr_scheduler, dataset, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batch_count = 0
    sample_count = 0
    target_squared_error_sum = torch.zeros(len(SURROGATE_TARGET_NAMES), device=device)
    target_absolute_error_sum = torch.zeros(len(SURROGATE_TARGET_NAMES), device=device)
    gradient_norms = []

    for coords, conditions, target in dataloader:
        coords = coords.to(device)
        conditions = conditions.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(coords, conditions)
        loss = criterion(pred, target)
        loss.backward()
        gradient_norms.append(compute_global_gradient_norm(model.parameters()))
        optimizer.step()
        lr_scheduler.step()

        with torch.no_grad():
            pred_real = dataset.denormalize_targets(pred)
            target_real = dataset.denormalize_targets(target)
            mae = torch.mean(torch.abs(pred_real - target_real))

        total_loss += loss.item()
        total_mae += mae.item()
        batch_count += 1
        sample_count += target.size(0)
        target_squared_error_sum += torch.sum((pred.detach() - target) ** 2, dim=0)
        target_absolute_error_sum += torch.sum(torch.abs(pred_real - target_real), dim=0)

    if batch_count == 0:
        raise ValueError('Training dataloader produced zero batches')

    return {
        'loss': total_loss / batch_count,
        'mae': total_mae / batch_count,
        'per_target_mse': target_squared_error_sum.div(sample_count).cpu(),
        'per_target_mae': target_absolute_error_sum.div(sample_count).cpu(),
        'gradient_norm_mean': float(np.mean(gradient_norms)),
        'gradient_norm_max': float(np.max(gradient_norms)),
    }


def compute_global_gradient_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += torch.sum(parameter.grad.detach() ** 2).item()
    return squared_norm ** 0.5


def initialize_training_metrics(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=TRAINING_METRIC_FIELDS).writeheader()


def append_training_metrics(path, metrics):
    missing_fields = set(TRAINING_METRIC_FIELDS) - set(metrics)
    extra_fields = set(metrics) - set(TRAINING_METRIC_FIELDS)
    if missing_fields or extra_fields:
        raise ValueError(
            f'Training metric fields do not match schema: missing={sorted(missing_fields)}, '
            f'extra={sorted(extra_fields)}'
        )
    with open(path, 'a', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=TRAINING_METRIC_FIELDS).writerow(metrics)


def build_epoch_metrics(epoch, train_result, val_result, optimizer):
    metrics = {
        'epoch': epoch + 1,
        'train_loss': train_result['loss'],
        'val_loss': val_result['loss'],
        'train_mae': train_result['mae'],
        'val_mae': val_result['mae'],
        'learning_rate': optimizer.param_groups[0]['lr'],
        'train_grad_norm_mean': train_result['gradient_norm_mean'],
        'train_grad_norm_max': train_result['gradient_norm_max'],
    }
    for split_name, result in [('train', train_result), ('val', val_result)]:
        for index, target_name in enumerate(SURROGATE_TARGET_NAMES):
            metrics[f'{split_name}_{target_name.lower()}_mse'] = float(result['per_target_mse'][index])
            metrics[f'{split_name}_{target_name.lower()}_mae'] = float(result['per_target_mae'][index])
    return metrics


def plot_training_curves(train_values, val_values, ylabel, title, path):
    epochs = np.arange(1, len(train_values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label='Train')
    plt.plot(epochs, val_values, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved plot to {path}")


def plot_prediction_scatter(targets, predictions, target_index, title, path):
    actual = targets[:, target_index].numpy()
    predicted = predictions[:, target_index].numpy()
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    margin = (max_value - min_value) * 0.05
    if margin == 0.0:
        margin = 1e-3

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, s=12, alpha=0.65)
    plt.plot(
        [min_value - margin, max_value + margin],
        [min_value - margin, max_value + margin],
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Perfect prediction',
    )
    plt.xlabel(f'Actual {title}')
    plt.ylabel(f'Predicted {title}')
    plt.title(f'Validation {title}: Actual vs Predicted')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved plot to {path}")


def save_checkpoint(model, best_model_path, norm_path, config, dataset, epoch, val_loss):
    path = best_model_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'target_names': dataset.target_names,
        'condition_names': dataset.condition_names,
        'target_loss_weights': config['surrogate_target_loss_weights'],
        'norm_path': norm_path,
    }, path)
    print(f"Saved best surrogate model to {path}")


def clone_state_dict(state_dict):
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def build_weighted_mse_loss(config, device):
    weights = torch.tensor(config['surrogate_target_loss_weights'], dtype=torch.float32, device=device)
    if weights.numel() != len(SURROGATE_TARGET_NAMES):
        raise ValueError(
            f"surrogate_target_loss_weights must contain "
            f"{len(SURROGATE_TARGET_NAMES)} values for {SURROGATE_TARGET_NAMES}, "
            f"got {weights.numel()}"
        )
    if torch.any(weights < 0):
        raise ValueError(f"surrogate_target_loss_weights must be non-negative, got {weights.tolist()}")
    if torch.sum(weights) <= 0:
        raise ValueError(f"At least one surrogate_target_loss_weights value must be positive, got {weights.tolist()}")
    return WeightedMSELoss(weights)


def build_surrogate_clr(config, optimizer, batches_per_epoch):
    missing_keys = [key for key in CLR_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f'Surrogate CLR configuration is missing keys: {missing_keys}')
    mode = config['surrogate_clr_mode']
    base_lr = float(config['surrogate_clr_base_lr'])
    max_lr = float(config['surrogate_clr_max_lr'])
    step_size_epochs = config['surrogate_clr_step_size_epochs']
    if mode != CLR_MODE:
        raise ValueError(f'surrogate_clr_mode must be {CLR_MODE}, got {mode}')
    if base_lr <= 0.0 or max_lr <= base_lr:
        raise ValueError(
            'Surrogate CLR bounds must satisfy 0 < base_lr < max_lr, '
            f'got {base_lr}, {max_lr}'
        )
    if not isinstance(step_size_epochs, int) or step_size_epochs <= 0:
        raise ValueError(
            'surrogate_clr_step_size_epochs must be a positive integer, '
            f'got {step_size_epochs}'
        )
    if batches_per_epoch <= 0:
        raise ValueError(f'batches_per_epoch must be positive, got {batches_per_epoch}')
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_epochs * batches_per_epoch,
        mode=mode,
        cycle_momentum=False,
    )


def train_surrogate_from_config(config, epochs_override=None, save_artifacts=True, trial=None):
    device = resolve_device(config)
    print(f"Using device: {device}")

    dataset_name, dataset_config = resolve_surrogate_dataset_config(config)
    dataset = AirfoilSurrogateDataset(
        dataset_config['data_path'],
        dataset_config['norm_path'],
        config,
        save_norm=save_artifacts,
    )
    train_set, val_set, test_set = split_dataset(dataset, config)
    print(
        f"Dataset '{dataset_name}' split: train={len(train_set)}, "
        f"validation={len(val_set)}, test={len(test_set)}"
    )

    batch_size = config['surrogate_batch_size']
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    model = AerodynamicSurrogate(config).to(device)
    # Weighted MSE loss on normalized targets:
    # L = (1 / (B * K)) * sum_b sum_k w[k] * (y_hat[b, k] - y[b, k])^2,
    # where target order is [CM, CL, CD], K = 3, and w comes from
    # surrogate_target_loss_weights.
    criterion = build_weighted_mse_loss(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['surrogate_clr_base_lr']),
        weight_decay=float(config['surrogate_weight_decay']),
    )
    lr_scheduler = build_surrogate_clr(config, optimizer, len(train_loader))

    epochs = config['surrogate_epochs']
    if epochs_override is not None:
        epochs = epochs_override

    train_losses = []
    val_losses = []
    train_errors = []
    val_errors = []
    best_val_loss = float('inf')
    best_val_mae = None
    best_epoch = None
    best_model_state = None

    if save_artifacts:
        initialize_training_metrics(TRAINING_METRICS_PATH)

    for epoch in range(epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            dataset,
            device,
        )
        val_result = evaluate(model, val_loader, criterion, dataset, device)

        train_losses.append(train_result['loss'])
        val_losses.append(val_result['loss'])
        train_errors.append(train_result['mae'])
        val_errors.append(val_result['mae'])

        if save_artifacts:
            append_training_metrics(
                TRAINING_METRICS_PATH,
                build_epoch_metrics(epoch, train_result, val_result, optimizer),
            )

        if val_result['loss'] < best_val_loss:
            best_val_loss = val_result['loss']
            best_val_mae = val_result['mae']
            best_epoch = epoch
            best_model_state = clone_state_dict(model.state_dict())

        if trial is not None:
            trial.report(val_result['loss'], epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"[Train loss: {train_result['loss']:.6f}] [Val loss: {val_result['loss']:.6f}] "
                f"[Train MAE: {train_result['mae']:.6f}] [Val MAE: {val_result['mae']:.6f}]"
            )

    if best_model_state is None:
        raise ValueError('No best model state was recorded during training')

    model.load_state_dict(best_model_state)

    if save_artifacts:
        save_checkpoint(
            model,
            dataset_config['best_model_path'],
            dataset_config['norm_path'],
            config,
            dataset,
            best_epoch,
            best_val_loss,
        )

        plot_training_curves(
            train_losses,
            val_losses,
            'MSE Loss',
            'Surrogate Training Loss',
            LOSS_PLOT_PATH,
        )
        plot_training_curves(
            train_errors,
            val_errors,
            'MAE',
            'Surrogate Prediction Error',
            ERROR_PLOT_PATH,
        )

        val_result = evaluate(model, val_loader, criterion, dataset, device)

        plot_prediction_scatter(
            val_result['targets'],
            val_result['predictions'],
            dataset.target_names.index('CL'),
            'CL',
            VALIDATION_PLOT_PATHS['CL'],
        )
        plot_prediction_scatter(
            val_result['targets'],
            val_result['predictions'],
            dataset.target_names.index('CD'),
            'CD',
            VALIDATION_PLOT_PATHS['CD'],
        )
        plot_prediction_scatter(
            val_result['targets'],
            val_result['predictions'],
            dataset.target_names.index('CM'),
            'CM',
            VALIDATION_PLOT_PATHS['CM'],
        )

    print(f"Best validation loss: {best_val_loss:.6f}")
    if best_val_mae is None:
        raise ValueError('No best validation MAE was recorded during training')
    return {
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_epoch': best_epoch,
    }


def run_training(config_path, epochs_override=None):
    config = load_config(config_path)
    return train_surrogate_from_config(
        config,
        epochs_override=epochs_override,
        save_artifacts=True,
        trial=None,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train airfoil aerodynamic surrogate model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--epochs', type=int, help='Override surrogate_epochs for smoke tests')
    args = parser.parse_args()
    run_training(args.config, epochs_override=args.epochs)
