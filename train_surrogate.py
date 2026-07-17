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
from surrogate_split import (
    build_fold_training_indices,
    load_cross_validation_manifest,
    resolve_surrogate_dataset_config,
)


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
TRAINING_METRIC_FIELDS = (
    ['epoch', 'train_loss', 'train_mae']
    + [f'train_{target.lower()}_mse' for target in SURROGATE_TARGET_NAMES]
    + [f'train_{target.lower()}_mae' for target in SURROGATE_TARGET_NAMES]
    + ['learning_rate', 'train_grad_norm_mean', 'train_grad_norm_max']
)


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights.float())

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(
                f'Prediction and target shapes must match, got {predictions.shape} and '
                f'{targets.shape}'
            )
        return torch.mean((predictions - targets) ** 2 * self.weights)


class AirfoilSurrogateDataset(Dataset):
    def __init__(self, raw_data, norm_state):
        if not raw_data:
            raise ValueError('Dataset is empty')
        self.label_names = DATASET_LABEL_NAMES
        self.cd_key = DATASET_CD_KEY
        self.condition_names = SURROGATE_CONDITION_NAMES
        self.target_names = SURROGATE_TARGET_NAMES
        self.condition_indices = [self.label_names.index(name) for name in self.condition_names]
        self.load_norm_state(norm_state)
        self.samples = [self.normalize_item(item) for item in raw_data]

    @classmethod
    def from_training_indices(cls, raw_data, training_indices):
        if not training_indices:
            raise ValueError('Normalization training indices are empty')
        template = cls.__new__(cls)
        template.label_names = DATASET_LABEL_NAMES
        template.cd_key = DATASET_CD_KEY
        template.condition_names = SURROGATE_CONDITION_NAMES
        template.target_names = SURROGATE_TARGET_NAMES
        template.condition_indices = [template.label_names.index(name) for name in template.condition_names]
        norm_state = template.build_norm_state(raw_data, training_indices)
        return cls(raw_data, norm_state), norm_state

    @classmethod
    def from_norm_path(cls, raw_data, norm_path):
        norm_state = torch.load(norm_path, weights_only=True)
        if norm_state['source_split'] != 'development':
            raise ValueError(
                f"Unexpected surrogate normalization source: {norm_state['source_split']}"
            )
        return cls(raw_data, norm_state)

    def load_norm_state(self, norm_state):
        required_sections = ('coord', 'condition', 'target')
        missing_sections = [name for name in required_sections if name not in norm_state]
        if missing_sections:
            raise ValueError(f'Normalization state is missing sections: {missing_sections}')
        if norm_state['condition']['names'] != self.condition_names:
            raise ValueError(
                f"Unexpected condition names: {norm_state['condition']['names']}"
            )
        if norm_state['target']['names'] != self.target_names:
            raise ValueError(f"Unexpected target names: {norm_state['target']['names']}")
        self.x_min = norm_state['coord']['x_min'].float()
        self.x_max = norm_state['coord']['x_max'].float()
        self.y_min = norm_state['coord']['y_min'].float()
        self.y_max = norm_state['coord']['y_max'].float()
        self.condition_mean = norm_state['condition']['mean'].float()
        self.condition_std = norm_state['condition']['std'].float()
        self.target_mean = norm_state['target']['mean'].float()
        self.target_std = norm_state['target']['std'].float()

    def build_norm_state(self, raw_data, training_indices):
        train_items = [raw_data[index] for index in training_indices]
        coords = torch.stack([item['x'] for item in train_items]).float().view(len(train_items), -1, 2)
        conditions = torch.stack([self.extract_conditions(item) for item in train_items]).float()
        targets = torch.stack([self.extract_targets(item) for item in train_items]).float()
        return {
            'coord': {
                'x_min': coords[:, :, 0].min(),
                'x_max': coords[:, :, 0].max(),
                'y_min': coords[:, :, 1].min(),
                'y_max': coords[:, :, 1].max(),
            },
            'condition': {
                'names': self.condition_names,
                'mean': conditions.mean(dim=0),
                'std': conditions.std(dim=0) + 1e-8,
            },
            'target': {
                'names': self.target_names,
                'mean': targets.mean(dim=0),
                'std': targets.std(dim=0) + 1e-8,
            },
        }

    def normalize_item(self, item):
        coords = item['x'].float().clone().view(-1, 2)
        coords[:, 0] = (coords[:, 0] - self.x_min) / (self.x_max - self.x_min + 1e-8)
        coords[:, 1] = (coords[:, 1] - self.y_min) / (self.y_max - self.y_min + 1e-8)
        condition = (self.extract_conditions(item) - self.condition_mean) / self.condition_std
        target = (self.extract_targets(item) - self.target_mean) / self.target_std
        return {'coords': coords.view(-1), 'conditions': condition, 'targets': target}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample['coords'], sample['conditions'], sample['targets']

    def extract_conditions(self, item):
        return item['y'][self.condition_indices].float()

    def extract_targets(self, item):
        values = []
        for name in self.target_names:
            values.append(item[self.cd_key] if name == 'CD' else item['y'][self.label_names.index(name)])
        return torch.tensor(values, dtype=torch.float32)

    def denormalize_targets(self, values):
        return values * self.target_std.to(values.device) + self.target_mean.to(values.device)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def resolve_device(config):
    device_cfg = config['device'].lower()
    if device_cfg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_cfg == 'cpu':
        return torch.device('cpu')
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(f"Unknown device configuration: {config['device']}")


def load_raw_data_and_manifest(config):
    dataset_config = resolve_surrogate_dataset_config(config)
    raw_data = torch.load(dataset_config['data_path'], weights_only=True)
    if not raw_data:
        raise ValueError(f"Dataset is empty: {dataset_config['data_path']}")
    return raw_data, load_cross_validation_manifest(raw_data, config)


def save_normalization_state(norm_state, config, training_indices):
    dataset_config = resolve_surrogate_dataset_config(config)
    manifest = load_cross_validation_manifest(
        torch.load(dataset_config['data_path'], weights_only=True),
        config,
    )
    saved_state = {
        'source_split': 'development',
        'split_seed': config['surrogate_seed'],
        'test_ratio': config['surrogate_test_ratio'],
        'fold_count': config['surrogate_cv_fold_count'],
        'normalization_sample_count': len(training_indices),
        'development_sample_count': len(manifest['development_indices']),
        **norm_state,
    }
    norm_path = dataset_config['norm_path']
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    torch.save(saved_state, norm_path)
    return norm_path


def make_loader(dataset, indices, config, shuffle):
    if not indices:
        raise ValueError('DataLoader indices are empty')
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config['surrogate_batch_size'],
        shuffle=shuffle,
        drop_last=False,
    )


def set_training_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            coords, conditions, target = coords.to(device), conditions.to(device), target.to(device)
            prediction = model(coords, conditions)
            loss = criterion(prediction, target)
            prediction_real = dataset.denormalize_targets(prediction)
            target_real = dataset.denormalize_targets(target)
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(prediction_real - target_real)).item()
            batch_count += 1
            sample_count += target.size(0)
            target_squared_error_sum += torch.sum((prediction - target) ** 2, dim=0)
            target_absolute_error_sum += torch.sum(torch.abs(prediction_real - target_real), dim=0)
            predictions.append(prediction_real.cpu())
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


def compute_global_gradient_norm(parameters):
    squared_norm = sum(
        torch.sum(parameter.grad.detach() ** 2).item()
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared_norm ** 0.5


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
        coords, conditions, target = coords.to(device), conditions.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(coords, conditions)
        loss = criterion(prediction, target)
        loss.backward()
        gradient_norms.append(compute_global_gradient_norm(model.parameters()))
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            prediction_real = dataset.denormalize_targets(prediction)
            target_real = dataset.denormalize_targets(target)
            total_mae += torch.mean(torch.abs(prediction_real - target_real)).item()
        total_loss += loss.item()
        batch_count += 1
        sample_count += target.size(0)
        target_squared_error_sum += torch.sum((prediction.detach() - target) ** 2, dim=0)
        target_absolute_error_sum += torch.sum(torch.abs(prediction_real - target_real), dim=0)
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


def build_weighted_mse_loss(config, device):
    weights = torch.tensor(config['surrogate_target_loss_weights'], dtype=torch.float32, device=device)
    if weights.numel() != len(SURROGATE_TARGET_NAMES):
        raise ValueError(
            f'surrogate_target_loss_weights must contain {len(SURROGATE_TARGET_NAMES)} '
            f'values for {SURROGATE_TARGET_NAMES}, got {weights.numel()}'
        )
    if torch.any(weights < 0) or torch.sum(weights) <= 0:
        raise ValueError(f'Invalid surrogate_target_loss_weights: {weights.tolist()}')
    return WeightedMSELoss(weights)


def build_surrogate_clr(config, optimizer, batches_per_epoch):
    missing_keys = [key for key in CLR_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f'Surrogate CLR configuration is missing keys: {missing_keys}')
    if config['surrogate_clr_mode'] != CLR_MODE:
        raise ValueError(f"surrogate_clr_mode must be {CLR_MODE}")
    base_lr = float(config['surrogate_clr_base_lr'])
    max_lr = float(config['surrogate_clr_max_lr'])
    step_size_epochs = config['surrogate_clr_step_size_epochs']
    if base_lr <= 0.0 or max_lr <= base_lr:
        raise ValueError('Surrogate CLR bounds must satisfy 0 < base_lr < max_lr')
    if not isinstance(step_size_epochs, int) or step_size_epochs <= 0:
        raise ValueError('surrogate_clr_step_size_epochs must be a positive integer')
    if batches_per_epoch <= 0:
        raise ValueError('batches_per_epoch must be positive')
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_epochs * batches_per_epoch,
        mode=CLR_MODE,
        cycle_momentum=False,
    )


def initialize_training_metrics(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as file:
        csv.DictWriter(file, fieldnames=TRAINING_METRIC_FIELDS).writeheader()


def append_training_metrics(path, metrics):
    if set(metrics) != set(TRAINING_METRIC_FIELDS):
        raise ValueError('Training metric fields do not match schema')
    with open(path, 'a', encoding='utf-8', newline='') as file:
        csv.DictWriter(file, fieldnames=TRAINING_METRIC_FIELDS).writerow(metrics)


def build_epoch_metrics(epoch, train_result, optimizer):
    metrics = {
        'epoch': epoch + 1,
        'train_loss': train_result['loss'],
        'train_mae': train_result['mae'],
        'learning_rate': optimizer.param_groups[0]['lr'],
        'train_grad_norm_mean': train_result['gradient_norm_mean'],
        'train_grad_norm_max': train_result['gradient_norm_max'],
    }
    for index, target_name in enumerate(SURROGATE_TARGET_NAMES):
        metrics[f'train_{target_name.lower()}_mse'] = float(train_result['per_target_mse'][index])
        metrics[f'train_{target_name.lower()}_mae'] = float(train_result['per_target_mae'][index])
    return metrics


def plot_training_curve(values, ylabel, title, path):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_prediction_scatter(targets, predictions, target_index, title, path):
    actual = targets[:, target_index].numpy()
    predicted = predictions[:, target_index].numpy()
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    margin = max((max_value - min_value) * 0.05, 1e-3)
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, s=12, alpha=0.65)
    plt.plot([min_value - margin, max_value + margin], [min_value - margin, max_value + margin], 'r--')
    plt.xlabel(f'Actual {title}')
    plt.ylabel(f'Predicted {title}')
    plt.title(f'{title}: Actual vs Predicted')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def train_fixed_epochs(
    config,
    dataset,
    train_indices,
    validation_indices,
    device,
    training_seed,
    record_metrics_path=None,
):
    set_training_seed(training_seed)
    train_loader = make_loader(dataset, train_indices, config, shuffle=True)
    validation_loader = make_loader(dataset, validation_indices, config, shuffle=False)
    model = AerodynamicSurrogate(config).to(device)
    criterion = build_weighted_mse_loss(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['surrogate_clr_base_lr']),
        weight_decay=float(config['surrogate_weight_decay']),
    )
    scheduler = build_surrogate_clr(config, optimizer, len(train_loader))
    epochs = config['surrogate_cv_epochs']
    if record_metrics_path is not None:
        initialize_training_metrics(record_metrics_path)
    train_losses = []
    train_errors = []
    for epoch in range(epochs):
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, dataset, device
        )
        train_losses.append(train_result['loss'])
        train_errors.append(train_result['mae'])
        if record_metrics_path is not None:
            append_training_metrics(record_metrics_path, build_epoch_metrics(epoch, train_result, optimizer))
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f'[Epoch {epoch + 1}/{epochs}] [Train loss: {train_result["loss"]:.6f}] '
                f'[Train MAE: {train_result["mae"]:.6f}]'
            )
    validation_result = evaluate(model, validation_loader, criterion, dataset, device)
    return model, validation_result, train_losses, train_errors


def run_cross_validation(config, trial=None):
    device = resolve_device(config)
    raw_data, manifest = load_raw_data_and_manifest(config)
    fold_results = []
    for fold_index, validation_indices in enumerate(manifest['fold_indices']):
        training_indices = build_fold_training_indices(manifest, fold_index)
        dataset, _ = AirfoilSurrogateDataset.from_training_indices(raw_data, training_indices)
        model, validation_result, _, _ = train_fixed_epochs(
            config,
            dataset,
            training_indices,
            validation_indices,
            device,
            config['surrogate_seed'] + fold_index,
        )
        fold_results.append(validation_result)
        running_loss = float(np.mean([result['loss'] for result in fold_results]))
        if trial is not None:
            trial.report(running_loss, fold_index)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        print(f'Fold {fold_index + 1}/{len(manifest["fold_indices"])} final validation loss: {validation_result["loss"]:.6f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    fold_losses = [result['loss'] for result in fold_results]
    return {
        'cv_loss_mean': float(np.mean(fold_losses)),
        'cv_loss_std': float(np.std(fold_losses)),
        'fold_losses': fold_losses,
        'fold_mae': [result['mae'] for result in fold_results],
        'fold_per_target_mae': [result['per_target_mae'].tolist() for result in fold_results],
    }


def save_final_checkpoint(model, config, norm_path):
    dataset_config = resolve_surrogate_dataset_config(config)
    checkpoint_path = dataset_config['best_model_path']
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_epoch_count': config['surrogate_cv_epochs'],
        'selection_policy': 'fixed_final_epoch',
        'target_names': SURROGATE_TARGET_NAMES,
        'condition_names': SURROGATE_CONDITION_NAMES,
        'target_loss_weights': config['surrogate_target_loss_weights'],
        'norm_path': norm_path,
    }, checkpoint_path)
    print(f'Saved final surrogate model to {checkpoint_path}')


def train_final_surrogate(config):
    device = resolve_device(config)
    raw_data, manifest = load_raw_data_and_manifest(config)
    development_indices = manifest['development_indices']
    dataset, norm_state = AirfoilSurrogateDataset.from_training_indices(raw_data, development_indices)
    norm_path = save_normalization_state(norm_state, config, development_indices)
    model, _, train_losses, train_errors = train_fixed_epochs(
        config,
        dataset,
        development_indices,
        development_indices,
        device,
        config['surrogate_seed'],
        record_metrics_path=TRAINING_METRICS_PATH,
    )
    save_final_checkpoint(model, config, norm_path)
    plot_training_curve(train_losses, 'MSE Loss', 'Final Surrogate Training Loss', LOSS_PLOT_PATH)
    plot_training_curve(train_errors, 'MAE', 'Final Surrogate Training Error', ERROR_PLOT_PATH)
    return {'training_epoch_count': config['surrogate_cv_epochs'], 'development_sample_count': len(development_indices)}


def run_training(config_path):
    return train_final_surrogate(load_config(config_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train final airfoil aerodynamic surrogate model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    args = parser.parse_args()
    run_training(args.config)
