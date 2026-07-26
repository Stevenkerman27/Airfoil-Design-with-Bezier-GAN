import copy
from unittest.mock import patch

import torch

from alternating_finetuning import (
    _set_actual_condition,
    build_stratum_candidate_requests,
    build_known_condition_records,
    resolve_alternating_config,
)
from gan_exploration import build_condition_record_strata
from reset_finetuning import existing_artifacts, reset_alternating_finetuning
from train_surrogate import load_config


def make_raw_data():
    return [
        {'y': torch.tensor([0.0, 100000.0, 0.2, -0.01])},
        {'y': torch.tensor([0.0, 100000.0, 0.4, -0.02])},
        {'y': torch.tensor([0.0, 100000.0, 0.6, -0.03])},
        {'y': torch.tensor([4.0, 100000.0, 0.8, -0.04])},
        {'y': torch.tensor([4.0, 100000.0, 1.0, -0.05])},
        {'y': torch.tensor([4.0, 100000.0, 1.2, -0.06])},
    ]


def test_successful_generated_records_extend_conditions_with_actual_xfoil_targets():
    generated = {
        'status': 'success',
        'cache_key': 'generated-key',
        'round_index': 0,
        'coords': torch.zeros(3, 2),
        'condition': torch.tensor([0.0, 100000.0, 9.0, 9.0]),
        'targets': torch.tensor([-0.12, 0.77, 0.014]),
    }
    _set_actual_condition(generated)

    records = build_known_condition_records(make_raw_data(), list(range(6)), [generated])

    generated_condition = records[-1]['condition']
    assert torch.allclose(
        generated_condition,
        torch.tensor([0.0, 100000.0, 0.77, -0.12]),
    )


def test_each_stratum_cycles_through_empirical_and_covariance_conditions():
    config = {'surrogate_seed': 19, 'noise_dimension': 3}
    alternating_config = {
        'existing_to_sampled_ratios': [[1, 1]],
        'mahalanobis_radii': [2.0],
        'noise_samples_per_condition': 2,
    }
    known_records = build_known_condition_records(make_raw_data(), list(range(6)), [])
    strata = build_condition_record_strata(known_records)
    stratum_key = (0.0, 100000.0)

    empirical_requests, empirical_sample = build_stratum_candidate_requests(
        config,
        alternating_config,
        round_index=0,
        stratum_index=0,
        stratum_key=stratum_key,
        stratum_records=strata[stratum_key],
        anchor_index=0,
        remaining_attempts=4,
    )
    covariance_requests, covariance_sample = build_stratum_candidate_requests(
        config,
        alternating_config,
        round_index=0,
        stratum_index=0,
        stratum_key=stratum_key,
        stratum_records=strata[stratum_key],
        anchor_index=1,
        remaining_attempts=4,
    )

    assert len(empirical_requests) == 2
    assert len(covariance_requests) == 2
    assert empirical_sample['source'] == 'empirical'
    assert covariance_sample['source'] == 'covariance'
    assert empirical_sample['mahalanobis_radius'] == 0.0
    assert covariance_sample['mahalanobis_radius'] <= 2.0
    assert all(request['stratum_key'] == stratum_key for request in covariance_requests)


def test_final_anchor_is_clipped_by_the_attempt_limit():
    config = {'surrogate_seed': 19, 'noise_dimension': 3}
    alternating_config = {
        'existing_to_sampled_ratios': [[1, 1]],
        'mahalanobis_radii': [2.0],
        'noise_samples_per_condition': 2,
    }
    known_records = build_known_condition_records(make_raw_data(), list(range(6)), [])
    strata = build_condition_record_strata(known_records)
    stratum_key = (0.0, 100000.0)
    requests, _sampled = build_stratum_candidate_requests(
        config,
        alternating_config,
        round_index=0,
        stratum_index=0,
        stratum_key=stratum_key,
        stratum_records=strata[stratum_key],
        anchor_index=2,
        remaining_attempts=1,
    )

    assert len(requests) == 1


def test_alternating_config_requires_one_radius_per_round():
    config = load_config('config.yaml')
    resolved = resolve_alternating_config(config)
    assert len(resolved['mahalanobis_radii']) == len(resolved['existing_to_sampled_ratios'])
    invalid = copy.deepcopy(config)
    radii = invalid['alternating_finetuning']['mahalanobis_radii']
    invalid['alternating_finetuning']['mahalanobis_radii'] = radii + [radii[-1]]

    try:
        resolve_alternating_config(invalid)
    except ValueError as error:
        assert 'must match existing_to_sampled_ratios length' in str(error)
    else:
        raise AssertionError('Expected mismatched alternating round lists to fail')


def test_alternating_config_requires_positive_collection_checkpoint_interval():
    config = load_config('config.yaml')
    invalid = copy.deepcopy(config)
    invalid['alternating_finetuning']['checkpoint_interval_collections'] = 0

    try:
        resolve_alternating_config(invalid)
    except ValueError as error:
        assert 'checkpoint_interval_collections must be a positive integer' in str(error)
    else:
        raise AssertionError('Expected invalid collection checkpoint interval to fail')


def test_reset_alternating_finetuning_removes_only_declared_artifacts():
    artifacts = (
        ('accumulated dataset', 'model/alternating_generated_dataset.pt', False),
        ('checkpoint directory', 'model/alternating_checkpoints', True),
        ('report directory', 'reports/alternating', True),
    )
    with (
        patch('reset_finetuning.existing_artifacts', return_value=list(artifacts)),
        patch('reset_finetuning.os.path.isfile', return_value=True),
        patch('reset_finetuning.os.path.isdir', return_value=True),
        patch('reset_finetuning.os.remove') as remove,
        patch('reset_finetuning.shutil.rmtree') as rmtree,
    ):
        deleted = reset_alternating_finetuning(artifacts)

    assert deleted == list(artifacts)
    remove.assert_called_once_with('model/alternating_generated_dataset.pt')
    assert rmtree.call_args_list == [
        (('model/alternating_checkpoints',), {}),
        (('reports/alternating',), {}),
    ]
