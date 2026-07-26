import torch

from surrogate_split import load_cross_validation_manifest


GAN_LABEL_ORDER = ['alpha', 'Re', 'CL', 'CM']
SURROGATE_DATASET_PATH = 'model/airfoil_dataset.pt'


def sample_development_conditions(raw_data, development_indices, count, seed):
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f'development sample count must be a positive integer, got {count}')
    if count > len(development_indices):
        raise ValueError(
            f'Requested {count} development conditions, only '
            f'{len(development_indices)} are available'
        )

    generator = torch.Generator().manual_seed(seed)
    selected_positions = torch.randperm(
        len(development_indices),
        generator=generator,
    )[:count].tolist()
    selected_conditions = []
    for position in selected_positions:
        dataset_index = development_indices[position]
        labels = raw_data[dataset_index]['y'].float()
        if labels.ndim != 1 or labels.numel() != len(GAN_LABEL_ORDER):
            raise ValueError(
                f'Dataset sample {dataset_index} must contain labels in order '
                f'{GAN_LABEL_ORDER}, got shape {tuple(labels.shape)}'
            )
        selected_conditions.append((dataset_index, labels.tolist()))
    return selected_conditions


def load_development_conditions(config, count):
    raw_data = torch.load(SURROGATE_DATASET_PATH, map_location='cpu', weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    return sample_development_conditions(
        raw_data,
        manifest['development_indices'],
        count,
        config['surrogate_seed'],
    )
