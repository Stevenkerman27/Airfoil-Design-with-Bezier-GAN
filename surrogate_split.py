import random

import torch


FOIL_ID_KEY = 'foil_id'
SPLIT_STRATEGY_AIRFOIL_GROUP = 'airfoil_group'
SPLIT_MANIFEST_VERSION = 2


def resolve_surrogate_dataset_config(config):
    if 'surrogate_dataset' not in config:
        raise ValueError('config is missing surrogate_dataset')
    dataset_config = config['surrogate_dataset']
    required_keys = ('data_path', 'split_path', 'norm_path', 'best_model_path')
    missing_keys = [key for key in required_keys if key not in dataset_config]
    if missing_keys:
        raise ValueError(f'surrogate_dataset is missing required keys: {missing_keys}')
    return dataset_config


def validate_cross_validation_config(config):
    test_ratio = config['surrogate_test_ratio']
    fold_count = config['surrogate_cv_fold_count']
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f'surrogate_test_ratio must be in (0, 1), got {test_ratio}')
    if not isinstance(fold_count, int) or fold_count <= 1:
        raise ValueError(
            f'surrogate_cv_fold_count must be an integer greater than 1, got {fold_count}'
        )


def collect_airfoil_groups(raw_data):
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
    return groups


def assign_groups_to_bins(group_items, bin_names, target_sizes, seed):
    if len(bin_names) != len(target_sizes):
        raise ValueError('bin_names and target_sizes must have the same length')
    if len(group_items) < len(bin_names):
        raise ValueError(
            f'Need at least {len(bin_names)} airfoil groups, got {len(group_items)}'
        )
    if any(size <= 0 for size in target_sizes):
        raise ValueError(f'All bin target sizes must be positive, got {target_sizes}')

    shuffled_items = list(group_items)
    random.Random(seed).shuffle(shuffled_items)
    shuffled_items.sort(key=lambda item: len(item[1]), reverse=True)

    assigned = {name: [] for name in bin_names}
    current_sizes = {name: 0 for name in bin_names}
    for foil_id, indices in shuffled_items:
        destination = max(
            bin_names,
            key=lambda name: (target_sizes[bin_names.index(name)] - current_sizes[name])
            / target_sizes[bin_names.index(name)],
        )
        assigned[destination].extend(indices)
        current_sizes[destination] += len(indices)

    empty_bins = [name for name in bin_names if not assigned[name]]
    if empty_bins:
        raise ValueError(f'Group assignment produced empty bins: {empty_bins}')
    return assigned


def build_cross_validation_manifest(raw_data, test_ratio, fold_count, seed):
    if not raw_data:
        raise ValueError('Cannot split an empty dataset')
    config = {
        'surrogate_test_ratio': test_ratio,
        'surrogate_cv_fold_count': fold_count,
    }
    validate_cross_validation_config(config)

    groups = collect_airfoil_groups(raw_data)
    if len(groups) < fold_count + 1:
        raise ValueError(
            f'Need at least {fold_count + 1} distinct foil_id values for a test set and '
            f'{fold_count} folds, got {len(groups)}'
        )

    dataset_size = len(raw_data)
    test_target_size = int(dataset_size * test_ratio)
    development_target_size = dataset_size - test_target_size
    if test_target_size <= 0 or development_target_size <= 0:
        raise ValueError(
            f'Invalid test/development target sizes: {test_target_size}, {development_target_size}'
        )

    outer_assignment = assign_groups_to_bins(
        list(groups.items()),
        ('development', 'test'),
        (development_target_size, test_target_size),
        seed,
    )
    development_indices = outer_assignment['development']
    test_indices = outer_assignment['test']
    development_groups = [
        item for item in groups.items() if item[0] not in {
            raw_data[index][FOIL_ID_KEY] for index in test_indices
        }
    ]
    fold_names = [f'fold_{index}' for index in range(fold_count)]
    fold_target_size = len(development_indices) / fold_count
    fold_assignment = assign_groups_to_bins(
        development_groups,
        fold_names,
        [fold_target_size] * fold_count,
        seed + 1,
    )
    fold_indices = [fold_assignment[name] for name in fold_names]

    manifest = {
        'version': SPLIT_MANIFEST_VERSION,
        'split_strategy': SPLIT_STRATEGY_AIRFOIL_GROUP,
        'dataset_size': dataset_size,
        'test_ratio': test_ratio,
        'fold_count': fold_count,
        'seed': seed,
        'development_indices': development_indices,
        'test_indices': test_indices,
        'fold_indices': fold_indices,
    }
    validate_cross_validation_manifest(raw_data, manifest)
    return manifest


def validate_cross_validation_manifest(raw_data, manifest):
    required_keys = (
        'version',
        'split_strategy',
        'dataset_size',
        'test_ratio',
        'fold_count',
        'seed',
        'development_indices',
        'test_indices',
        'fold_indices',
    )
    missing_keys = [key for key in required_keys if key not in manifest]
    if missing_keys:
        raise ValueError(f'Split manifest is missing required keys: {missing_keys}')
    if manifest['version'] != SPLIT_MANIFEST_VERSION:
        raise ValueError(
            f"Unexpected split manifest version: {manifest['version']}"
        )
    if manifest['split_strategy'] != SPLIT_STRATEGY_AIRFOIL_GROUP:
        raise ValueError(
            f"split_strategy must be {SPLIT_STRATEGY_AIRFOIL_GROUP}, "
            f"got {manifest['split_strategy']}"
        )
    if manifest['dataset_size'] != len(raw_data):
        raise ValueError(
            f"Split manifest dataset size mismatch: expected {len(raw_data)}, "
            f"got {manifest['dataset_size']}"
        )

    config = {
        'surrogate_test_ratio': manifest['test_ratio'],
        'surrogate_cv_fold_count': manifest['fold_count'],
    }
    validate_cross_validation_config(config)
    development_indices = manifest['development_indices']
    test_indices = manifest['test_indices']
    fold_indices = manifest['fold_indices']
    if len(fold_indices) != manifest['fold_count']:
        raise ValueError(
            f"Expected {manifest['fold_count']} folds, got {len(fold_indices)}"
        )
    partitions = [development_indices, test_indices, *fold_indices]
    for partition_index, indices in enumerate(partitions):
        if not indices:
            raise ValueError(f'Split partition {partition_index} is empty')
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f'Split partition {partition_index} contains a non-integer index')
        if any(index < 0 or index >= len(raw_data) for index in indices):
            raise ValueError(f'Split partition {partition_index} contains an out-of-range index')

    all_outer_indices = development_indices + test_indices
    if len(all_outer_indices) != len(raw_data) or len(set(all_outer_indices)) != len(raw_data):
        raise ValueError('Development and test indices must cover each dataset item exactly once')
    all_fold_indices = [index for fold in fold_indices for index in fold]
    if sorted(all_fold_indices) != sorted(development_indices):
        raise ValueError('Cross-validation folds must cover the development indices exactly once')

    test_foil_ids = {raw_data[index][FOIL_ID_KEY] for index in test_indices}
    development_foil_ids = {raw_data[index][FOIL_ID_KEY] for index in development_indices}
    if test_foil_ids & development_foil_ids:
        raise ValueError('foil_id values leak between development and test sets')
    fold_foil_ids = [
        {raw_data[index][FOIL_ID_KEY] for index in fold}
        for fold in fold_indices
    ]
    for first_index, first_ids in enumerate(fold_foil_ids):
        for second_ids in fold_foil_ids[first_index + 1:]:
            if first_ids & second_ids:
                raise ValueError('foil_id values leak between cross-validation folds')


def load_cross_validation_manifest(raw_data, config):
    validate_cross_validation_config(config)
    dataset_config = resolve_surrogate_dataset_config(config)
    manifest_path = dataset_config['split_path']
    manifest = torch.load(manifest_path, weights_only=True)
    validate_cross_validation_manifest(raw_data, manifest)
    expected_values = {
        'test_ratio': config['surrogate_test_ratio'],
        'fold_count': config['surrogate_cv_fold_count'],
        'seed': config['surrogate_seed'],
    }
    for key, expected_value in expected_values.items():
        if manifest[key] != expected_value:
            raise ValueError(
                f"Split manifest {key} mismatch in {manifest_path}: expected "
                f"{expected_value}, got {manifest[key]}"
            )
    return manifest


def build_fold_training_indices(manifest, fold_index):
    fold_indices = manifest['fold_indices']
    if not isinstance(fold_index, int) or not 0 <= fold_index < len(fold_indices):
        raise ValueError(f'Invalid fold index {fold_index}')
    return [
        index
        for index, fold in enumerate(fold_indices)
        if index != fold_index
        for index in fold
    ]
