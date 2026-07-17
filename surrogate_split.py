import random

import torch


FOIL_ID_KEY = 'foil_id'
SPLIT_NAMES = ('train', 'val', 'test')
SPLIT_STRATEGY_RANDOM_SAMPLE = 'random_sample'
SPLIT_STRATEGY_AIRFOIL_GROUP = 'airfoil_group'
SPLIT_MANIFEST_VERSION = 1


def resolve_surrogate_dataset_config(config):
    dataset_name = config['surrogate_dataset_name']
    datasets = config['surrogate_datasets']
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown surrogate_dataset_name '{dataset_name}', "
            f"expected one of {list(datasets)}"
        )

    dataset_config = datasets[dataset_name]
    required_keys = (
        'data_path',
        'split_path',
        'split_strategy',
        'norm_path',
        'best_model_path',
    )
    missing_keys = [key for key in required_keys if key not in dataset_config]
    if missing_keys:
        raise ValueError(
            f"surrogate_datasets.{dataset_name} is missing required keys: {missing_keys}"
        )

    strategy = dataset_config['split_strategy']
    valid_strategies = (SPLIT_STRATEGY_RANDOM_SAMPLE, SPLIT_STRATEGY_AIRFOIL_GROUP)
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown split strategy '{strategy}', expected one of {valid_strategies}"
        )
    return dataset_name, dataset_config


def validate_split_ratio(split_ratio):
    if len(split_ratio) != len(SPLIT_NAMES):
        raise ValueError(
            f'surrogate_split_ratio must contain {len(SPLIT_NAMES)} values, got {split_ratio}'
        )
    total_ratio = sum(split_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f'surrogate_split_ratio must sum to 1.0, got {total_ratio}')
    if any(ratio <= 0 for ratio in split_ratio):
        raise ValueError(f'surrogate_split_ratio values must be positive, got {split_ratio}')


def build_split_sizes(dataset_size, split_ratio):
    validate_split_ratio(split_ratio)
    train_size = int(dataset_size * split_ratio[0])
    val_size = int(dataset_size * split_ratio[1])
    test_size = dataset_size - train_size - val_size
    sizes = (train_size, val_size, test_size)
    if min(sizes) <= 0:
        raise ValueError(f'Invalid split sizes for dataset size {dataset_size}: {sizes}')
    return sizes


def build_random_sample_split_indices(dataset_size, split_ratio, seed):
    train_size, val_size, _ = build_split_sizes(dataset_size, split_ratio)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return {
        'train': indices[:train_size],
        'val': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:],
    }


def build_airfoil_group_split_indices(raw_data, split_ratio, seed):
    dataset_size = len(raw_data)
    target_sizes = build_split_sizes(dataset_size, split_ratio)
    groups = {}
    for index, item in enumerate(raw_data):
        if FOIL_ID_KEY not in item:
            raise ValueError(
                f"Dataset item {index} is missing required '{FOIL_ID_KEY}'; "
                'regenerate the dataset with prepare_dataset.py'
            )
        foil_id = item[FOIL_ID_KEY]
        if not isinstance(foil_id, str) or not foil_id:
            raise ValueError(f"Dataset item {index} has invalid '{FOIL_ID_KEY}': {foil_id!r}")
        groups.setdefault(foil_id, []).append(index)

    if len(groups) < len(SPLIT_NAMES):
        raise ValueError(
            f'At least {len(SPLIT_NAMES)} distinct foil_id values are required, got {len(groups)}'
        )

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_indices = {name: [] for name in SPLIT_NAMES}
    current_sizes = [0] * len(SPLIT_NAMES)
    for foil_id, indices in group_items:
        group_size = len(indices)
        split_index = max(
            range(len(SPLIT_NAMES)),
            key=lambda index: (target_sizes[index] - current_sizes[index]) / target_sizes[index],
        )
        split_indices[SPLIT_NAMES[split_index]].extend(indices)
        current_sizes[split_index] += group_size

    validate_split_indices(raw_data, split_indices, SPLIT_STRATEGY_AIRFOIL_GROUP)
    return split_indices


def build_split_manifest(raw_data, split_ratio, seed, strategy):
    if strategy == SPLIT_STRATEGY_RANDOM_SAMPLE:
        split_indices = build_random_sample_split_indices(len(raw_data), split_ratio, seed)
    elif strategy == SPLIT_STRATEGY_AIRFOIL_GROUP:
        split_indices = build_airfoil_group_split_indices(raw_data, split_ratio, seed)
    else:
        raise ValueError(f'Unknown split strategy: {strategy}')

    validate_split_indices(raw_data, split_indices, strategy)
    return {
        'version': SPLIT_MANIFEST_VERSION,
        'split_strategy': strategy,
        'dataset_size': len(raw_data),
        'split_ratio': list(split_ratio),
        'seed': seed,
        'split_indices': split_indices,
    }


def validate_split_indices(raw_data, split_indices, strategy):
    dataset_size = len(raw_data)
    if tuple(split_indices) != SPLIT_NAMES:
        raise ValueError(
            f'Split names must be {SPLIT_NAMES}, got {tuple(split_indices)}'
        )

    all_indices = []
    for name in SPLIT_NAMES:
        indices = split_indices[name]
        if len(indices) == 0:
            raise ValueError(f'Split {name} is empty')
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f'Split {name} contains a non-integer index')
        if any(index < 0 or index >= dataset_size for index in indices):
            raise ValueError(f'Split {name} contains an out-of-range index')
        all_indices.extend(indices)

    if len(all_indices) != dataset_size or len(set(all_indices)) != dataset_size:
        raise ValueError('Split indices must cover each dataset item exactly once')

    if strategy == SPLIT_STRATEGY_AIRFOIL_GROUP:
        foil_ids_by_split = {
            name: {raw_data[index][FOIL_ID_KEY] for index in split_indices[name]}
            for name in SPLIT_NAMES
        }
        for first_index, first_name in enumerate(SPLIT_NAMES):
            for second_name in SPLIT_NAMES[first_index + 1:]:
                overlap = foil_ids_by_split[first_name] & foil_ids_by_split[second_name]
                if overlap:
                    raise ValueError(
                        f'Grouped split leaks foil_id values between {first_name} and '
                        f'{second_name}: {sorted(overlap)[:5]}'
                    )
    elif strategy != SPLIT_STRATEGY_RANDOM_SAMPLE:
        raise ValueError(f'Unknown split strategy: {strategy}')


def load_split_indices(raw_data, config):
    dataset_name, dataset_config = resolve_surrogate_dataset_config(config)
    manifest_path = dataset_config['split_path']
    manifest = torch.load(manifest_path, weights_only=True)
    expected_strategy = dataset_config['split_strategy']

    if manifest['version'] != SPLIT_MANIFEST_VERSION:
        raise ValueError(
            f"Unexpected split manifest version in {manifest_path}: {manifest['version']}"
        )
    if manifest['split_strategy'] != expected_strategy:
        raise ValueError(
            f"Split manifest strategy mismatch in {manifest_path}: "
            f"expected {expected_strategy}, got {manifest['split_strategy']}"
        )
    if manifest['dataset_size'] != len(raw_data):
        raise ValueError(
            f"Split manifest dataset size mismatch in {manifest_path}: "
            f"expected {len(raw_data)}, got {manifest['dataset_size']}"
        )
    if manifest['split_ratio'] != list(config['surrogate_split_ratio']):
        raise ValueError(f"Split manifest ratio mismatch in {manifest_path}")
    if manifest['seed'] != config['surrogate_seed']:
        raise ValueError(f"Split manifest seed mismatch in {manifest_path}")

    split_indices = manifest['split_indices']
    validate_split_indices(raw_data, split_indices, expected_strategy)
    return dataset_name, split_indices
