from unittest.mock import patch

import torch

from surrogate_split import (
    build_cross_validation_manifest,
    build_fold_training_indices,
    load_cross_validation_manifest,
)


def make_raw_data():
    raw_data = []
    for foil_index in range(12):
        for condition_index in range(5):
            raw_data.append({
                'x': torch.zeros(8),
                'y': torch.tensor([condition_index, 100000.0, 0.5, -0.05]),
                'cd': 0.01,
                'foil_id': f'foil_{foil_index}',
            })
    return raw_data


def build_config():
    return {
        'surrogate_test_ratio': 0.1,
        'surrogate_cv_fold_count': 5,
        'surrogate_seed': 20260704,
    }


def test_airfoil_group_cross_validation_does_not_leak_foil_ids():
    raw_data = make_raw_data()
    manifest = build_cross_validation_manifest(raw_data, 0.1, 5, 20260704)

    test_foil_ids = {raw_data[index]['foil_id'] for index in manifest['test_indices']}
    development_foil_ids = {
        raw_data[index]['foil_id'] for index in manifest['development_indices']
    }
    assert not test_foil_ids & development_foil_ids

    fold_foil_ids = [
        {raw_data[index]['foil_id'] for index in fold}
        for fold in manifest['fold_indices']
    ]
    for first_index, first_ids in enumerate(fold_foil_ids):
        for second_ids in fold_foil_ids[first_index + 1:]:
            assert not first_ids & second_ids

    assert sorted(index for fold in manifest['fold_indices'] for index in fold) == sorted(
        manifest['development_indices']
    )


def test_fold_training_indices_exclude_its_validation_fold():
    raw_data = make_raw_data()
    manifest = build_cross_validation_manifest(raw_data, 0.1, 5, 20260704)
    validation_indices = set(manifest['fold_indices'][2])
    training_indices = set(build_fold_training_indices(manifest, 2))

    assert not training_indices & validation_indices
    assert training_indices | validation_indices == set(manifest['development_indices'])
    assert not training_indices & set(manifest['test_indices'])


def test_cross_validation_manifest_matches_selected_configuration():
    raw_data = make_raw_data()
    manifest = build_cross_validation_manifest(raw_data, 0.1, 5, 20260704)

    with patch('surrogate_split.torch.load', return_value=manifest):
        loaded_manifest = load_cross_validation_manifest(raw_data, build_config())

    assert loaded_manifest['fold_count'] == 5
    assert len(loaded_manifest['test_indices']) > 0
