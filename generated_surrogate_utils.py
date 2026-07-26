import torch

from eval_surrogate import compute_target_metrics
from train import build_surrogate_conditions, normalize_surrogate_coords
from train_surrogate import evaluate


class GeneratedSurrogateDataset:
    def __init__(self, records, auxiliary_stats, device):
        if not records:
            raise ValueError('Generated surrogate dataset is empty')
        self.device = device
        self.target_mean = auxiliary_stats['surrogate_target_mean']
        self.target_std = auxiliary_stats['surrogate_target_std']
        physical_coords = torch.stack([record['coords'] for record in records]).to(device)
        physical_conditions = torch.stack([record['condition'] for record in records]).to(device)
        physical_targets = torch.stack([record['targets'] for record in records]).to(device)
        self.coords = normalize_surrogate_coords(
            physical_coords, auxiliary_stats['surrogate_coord']
        )
        self.conditions = build_surrogate_conditions(physical_conditions, auxiliary_stats)
        self.targets = (physical_targets - self.target_mean) / self.target_std

    def __len__(self):
        return self.coords.size(0)

    def prepare_indices(self, indices):
        prepared = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if prepared.ndim != 1 or prepared.numel() == 0:
            raise ValueError('Generated batch indices must be a non-empty one-dimensional sequence')
        if torch.any(prepared < 0) or torch.any(prepared >= len(self)):
            raise ValueError('Generated batch indices contain an out-of-range value')
        return prepared

    def batch_count(self, indices, batch_size):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('Generated batch size must be a positive integer')
        return (indices.numel() + batch_size - 1) // batch_size

    def iter_batches(self, indices, batch_size, shuffle):
        if indices.device != self.coords.device:
            raise ValueError('Generated batch indices must reside on the dataset device')
        if indices.dtype != torch.long:
            raise ValueError('Generated batch indices must use torch.long dtype')
        if shuffle:
            indices = indices[torch.randperm(indices.numel(), device=self.coords.device)]
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield (
                self.coords[batch_indices],
                self.conditions[batch_indices],
                self.targets[batch_indices],
            )

    def denormalize_targets(self, values):
        return values * self.target_std + self.target_mean


def evaluate_generated_surrogate_metrics(model, dataset, indices, criterion, batch_size, device):
    result = evaluate(model, dataset, indices, criterion, batch_size, device)
    metrics = {
        'weighted_mse': float(result['loss']),
        'mae': float(result['mae']),
    }
    metrics.update(
        compute_target_metrics(
            result['predictions'], result['targets'], ['CM', 'CL', 'CD']
        )
    )
    return metrics
