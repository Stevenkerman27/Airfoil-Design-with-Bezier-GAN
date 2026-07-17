import torch
from unittest.mock import patch

from surrogate_split import (
    SPLIT_STRATEGY_AIRFOIL_GROUP,
    SPLIT_STRATEGY_RANDOM_SAMPLE,
    build_split_manifest,
    load_split_indices,
)


def make_raw_data():
    raw_data = []
    for foil_index in range(12):
        for condition_index in range(5):
            raw_data.append(
                {
                    'x': torch.zeros(8),
                    'y': torch.tensor([condition_index, 100000.0, 0.5, 0.1, -0.05]),
                    'cd': 0.01,
                    'foil_id': f'foil_{foil_index}',
                }
            )
    return raw_data


def test_airfoil_group_split_does_not_leak_foil_ids():
    raw_data = make_raw_data()
    manifest = build_split_manifest(
        raw_data,
        [0.8, 0.1, 0.1],
        20260704,
        SPLIT_STRATEGY_AIRFOIL_GROUP,
    )

    split_foil_ids = {
        name: {raw_data[index]['foil_id'] for index in indices}
        for name, indices in manifest['split_indices'].items()
    }
    assert len(split_foil_ids['train'] & split_foil_ids['val']) == 0
    assert len(split_foil_ids['train'] & split_foil_ids['test']) == 0
    assert len(split_foil_ids['val'] & split_foil_ids['test']) == 0
    assert sum(len(indices) for indices in manifest['split_indices'].values()) == len(raw_data)


def test_selected_split_manifest_is_loaded():
    raw_data = make_raw_data()
    group_manifest = build_split_manifest(
        raw_data,
        [0.8, 0.1, 0.1],
        20260704,
        SPLIT_STRATEGY_AIRFOIL_GROUP,
    )
    config = {
        'surrogate_dataset_name': 'airfoil_group',
        'surrogate_datasets': {
            'random_sample': {
                'data_path': 'unused.pt',
                'split_path': 'random.pt',
                'split_strategy': SPLIT_STRATEGY_RANDOM_SAMPLE,
                'norm_path': 'random_norm.pt',
                'best_model_path': 'random_best.pt',
            },
            'airfoil_group': {
                'data_path': 'unused.pt',
                'split_path': 'group.pt',
                'split_strategy': SPLIT_STRATEGY_AIRFOIL_GROUP,
                'norm_path': 'group_norm.pt',
                'best_model_path': 'group_best.pt',
            },
        },
        'surrogate_split_ratio': [0.8, 0.1, 0.1],
        'surrogate_seed': 20260704,
    }

    with patch('surrogate_split.torch.load', return_value=group_manifest):
        dataset_name, split_indices = load_split_indices(raw_data, config)

    assert dataset_name == 'airfoil_group'
    assert sum(len(indices) for indices in split_indices.values()) == len(raw_data)
