import csv
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml
from optuna.trial import FixedTrial


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import AerodynamicSurrogate
from optimize_surrogate import apply_trial_params
from lr_range_test import (
    aggregate_smoothed_losses,
    exponential_learning_rate,
    update_ema,
    validate_lr_range_config,
)
from train_surrogate import (
    TRAINING_METRIC_FIELDS,
    append_training_metrics,
    build_epoch_metrics,
    build_surrogate_clr,
    initialize_training_metrics,
)


def build_config():
    return {
        'num_output_points': 100,
        'surrogate_cond_dim': 2,
        'surrogate_out_dim': 3,
        'surrogate_hid_node': 32,
        'surrogate_hid_layer': 2,
        'surrogate_conv1_channels': 8,
        'surrogate_conv1_kernel': 7,
        'surrogate_conv2_channels': 4,
        'surrogate_conv2_kernel': 11,
        'surrogate_conv2_stride': 3,
    }


def test_surrogate_forward_with_independent_convolution_config():
    model = AerodynamicSurrogate(build_config())
    coords = torch.randn(4, 200)
    conditions = torch.randn(4, 2)

    output = model(coords, conditions)

    assert output.shape == (4, 3)


def test_surrogate_rejects_even_kernel_size():
    config = build_config()
    config['surrogate_conv1_kernel'] = 8

    with pytest.raises(ValueError, match='surrogate_conv1_kernel must be a positive odd integer'):
        AerodynamicSurrogate(config)


def test_surrogate_optuna_kernel_search_space():
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    search_space = config['optuna']['search_space']
    expected_kernels = [3, 5, 7, 9, 11, 15]

    assert search_space['surrogate_conv1_kernel'] == expected_kernels
    assert search_space['surrogate_conv2_kernel'] == expected_kernels
    assert 'disc_conv_kernel' not in search_space
    assert 'disc_conv2_kernel' not in search_space


def test_optuna_trial_configures_both_surrogate_kernels():
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    trial = FixedTrial(
        {
            'surrogate_weight_decay': 1e-6,
            'surrogate_batch_size': 16,
            'surrogate_hid_node': 64,
            'surrogate_hid_layer': 1,
            'surrogate_conv1_channels': 4,
            'surrogate_conv1_kernel': 5,
            'surrogate_conv2_channels': 4,
            'surrogate_conv2_kernel': 9,
            'surrogate_conv2_stride': 2,
        }
    )

    trial_config = apply_trial_params(config, trial)

    assert trial_config['surrogate_conv1_kernel'] == 5
    assert trial_config['surrogate_conv2_kernel'] == 9
    AerodynamicSurrogate(trial_config)


def test_training_metrics_csv_uses_defined_schema(tmp_path):
    train_result = {
        'loss': 0.1,
        'mae': 0.01,
        'per_target_mse': torch.tensor([0.11, 0.12, 0.13]),
        'per_target_mae': torch.tensor([0.011, 0.012, 0.013]),
        'gradient_norm_mean': 0.5,
        'gradient_norm_max': 0.8,
    }
    optimizer = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=1e-3)
    metrics = build_epoch_metrics(2, train_result, optimizer)
    path = tmp_path / 'training_metrics.csv'

    initialize_training_metrics(path)
    append_training_metrics(path, metrics)

    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))

    assert tuple(rows[0]) == tuple(TRAINING_METRIC_FIELDS)
    assert len(rows) == 1
    assert rows[0]['epoch'] == '3'
    assert float(rows[0]['train_cd_mse']) == pytest.approx(0.13)
    assert float(rows[0]['train_grad_norm_max']) == pytest.approx(0.8)


def test_lr_range_schedule_reaches_configured_endpoints():
    assert exponential_learning_rate(1e-7, 1e-2, 0, 5) == pytest.approx(1e-7)
    assert exponential_learning_rate(1e-7, 1e-2, 4, 5) == pytest.approx(1e-2)


def test_lr_range_ema_is_bias_corrected_on_first_step():
    _, smoothed_loss = update_ema(0.0, 3.0, 0.98, 0)

    assert smoothed_loss == pytest.approx(3.0)


def test_lr_range_config_rejects_invalid_bounds():
    config = {
        'surrogate_lr_range_start': 1e-3,
        'surrogate_lr_range_end': 1e-4,
        'surrogate_lr_range_epochs': 1,
        'surrogate_lr_range_runs': 5,
        'surrogate_lr_range_ema_beta': 0.98,
        'surrogate_lr_range_max_loss_multiplier': 4.0,
    }

    with pytest.raises(ValueError, match='0 < start < end'):
        validate_lr_range_config(config)


def test_lr_range_aggregate_uses_available_runs_after_early_stop():
    records_by_run = [
        [
            {'learning_rate': 1e-4, 'smoothed_loss': 2.0},
            {'learning_rate': 2e-4, 'smoothed_loss': 1.0},
        ],
        [
            {'learning_rate': 1e-4, 'smoothed_loss': 4.0},
        ],
    ]

    learning_rates, mean_losses = aggregate_smoothed_losses(records_by_run)

    assert learning_rates == [1e-4, 2e-4]
    assert mean_losses == pytest.approx([3.0, 1.0])


def test_surrogate_clr_uses_epoch_based_half_cycle_and_adam_compatibility():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    config = {
        'surrogate_clr_mode': 'triangular2',
        'surrogate_clr_base_lr': 3e-4,
        'surrogate_clr_max_lr': 2e-3,
        'surrogate_clr_step_size_epochs': 4,
    }

    scheduler = build_surrogate_clr(config, optimizer, batches_per_epoch=11)

    assert scheduler.total_size == 88.0
    assert scheduler.step_ratio == 0.5
    assert scheduler.cycle_momentum is False
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3e-4)
