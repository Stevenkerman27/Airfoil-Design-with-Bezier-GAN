import torch
from unittest.mock import patch

from gan_exploration import (
    build_condition_statistics,
    build_development_condition_strata,
    resolve_exploration_config,
    sample_extended_conditions,
    optimize_latent_noise,
)


def make_raw_data():
    labels = [
        [0.0, 100000.0, 0.0, 1.0],
        [0.0, 100000.0, 1.0, 0.0],
        [0.0, 100000.0, 2.0, -1.0],
        [4.0, 100000.0, 10.0, 11.0],
        [4.0, 100000.0, 11.0, 10.0],
        [4.0, 100000.0, 12.0, 9.0],
    ]
    return [{'y': torch.tensor(value)} for value in labels]


def test_condition_statistics_distinguish_global_and_within_stratum_relationships():
    raw_data = make_raw_data()
    strata = build_development_condition_strata(raw_data, list(range(len(raw_data))))

    statistics = build_condition_statistics(strata)

    assert statistics['global_CL_CM_correlation'] > 0.9
    assert statistics['within_stratum_CL_CM_correlation'] < -0.9
    assert statistics['stratum_count'] == 2


def test_covariance_condition_sampling_keeps_cl_and_cm_jointly_bounded():
    raw_data = make_raw_data()
    samples, _ = sample_extended_conditions(
        raw_data,
        list(range(len(raw_data))),
        condition_count=40,
        empirical_condition_fraction=0.0,
        mahalanobis_radius=2.0,
        seed=17,
    )

    assert len(samples) == 40
    assert {sample['source'] for sample in samples} == {'covariance'}
    assert all(sample['mahalanobis_radius'] <= 2.0 for sample in samples)
    assert all(float(sample['condition'][0]) in {0.0, 4.0} for sample in samples)
    assert all(float(sample['condition'][1]) == 100000.0 for sample in samples)


def test_exploration_config_rejects_missing_shared_sampling_setting():
    config = {
        'gan_exploration': {
        }
    }

    try:
        resolve_exploration_config(config)
    except ValueError as error:
        assert 'missing required keys' in str(error)
    else:
        raise AssertionError('Expected incomplete exploration config to fail fast')


def test_latent_optimization_projects_noise_to_the_configured_trust_region():
    config = {
        'noise_dimension': 3,
        'gan_exploration': {
            'condition_count': 1,
            'empirical_condition_fraction': 1.0,
            'mahalanobis_radius': 2.0,
            'noise_samples_per_condition': 1,
            'generation_batch_size': 1,
            'xfoil_timeout_seconds': 1,
            'latent_optimization_start_count': 4,
            'latent_optimization_steps': 10,
            'latent_optimization_learning_rate': 1.0,
            'latent_trust_radius': 0.2,
            'cl_tolerance': 0.1,
            'cm_tolerance': 0.1,
            'cl_penalty_weight': 1.0,
            'cm_penalty_weight': 1.0,
        },
    }
    model = torch.nn.Linear(1, 1)
    condition = torch.tensor([2.0, 200000.0, 0.0, 0.0])

    def fake_generate_and_predict(_, __, ___, ____, noise, conditions):
        predictions = torch.stack([
            torch.zeros_like(noise[:, 0]),
            torch.zeros_like(noise[:, 0]),
            noise[:, 0],
        ], dim=1)
        coords = torch.zeros(noise.size(0), 3, 2, device=noise.device)
        return coords, predictions

    with patch('gan_exploration.generate_and_predict', side_effect=fake_generate_and_predict):
        result = optimize_latent_noise(
            model,
            model,
            {},
            config,
            condition,
            seed=3,
        )

    delta = result['optimized_noise'] - result['initial_noise']
    assert torch.all(torch.linalg.vector_norm(delta, dim=1) <= 0.200001)
