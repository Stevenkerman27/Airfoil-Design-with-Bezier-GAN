import os
import sys
from pathlib import Path

import torch
import yaml


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import GAN_LABEL_ORDER, build_gan_surrogate_target_weights


def test_gan_condition_and_target_weights_are_four_dimensional():
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    assert config['cond_dim'] == 4
    assert GAN_LABEL_ORDER == ['alpha', 'Re', 'CL', 'CM']
    assert 'gan_thickness_loss_weight' not in config
    assert 'gan_crossing_loss_weight' not in config
    assert config['gan_trailing_edge_crossing_point_count'] == 3
    assert config['gan_trailing_edge_crossing_te_weight'] == 2.0
    assert build_gan_surrogate_target_weights(config, torch.device('cpu')).tolist() == (
        config['gan_surrogate_target_loss_weights']
    )
