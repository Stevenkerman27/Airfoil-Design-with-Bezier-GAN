import copy
import os
import sys

import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encode_dat import fit_cst_airfoil_batch
from train_surrogate import load_config


def test_cst_batch_encoding_preserves_leading_edge_and_bounds():
    config = copy.deepcopy(load_config('config.yaml'))
    config['device'] = 'cpu'
    config['cst_encode']['iterations'] = 3
    target_path = 'foildata/processed_foil/ag03.dat'

    result = fit_cst_airfoil_batch([target_path], config, torch.device('cpu'))

    curve = result['curve']
    parameters = result['parameters']
    assert curve.shape == (1, config['num_output_points'], 2)
    assert torch.isfinite(curve).all()
    assert torch.equal(curve[:, 50, :], torch.zeros(1, 2))
    coefficient_count = config['cst']['shape_coefficient_count']
    assert parameters['upper_coefficients'].shape == (1, coefficient_count)
    assert parameters['lower_coefficients'].shape == (1, coefficient_count)
    assert torch.all((parameters['n1'] >= 0.25) & (parameters['n1'] <= 1.0))
    assert torch.all((parameters['n2'] >= 0.5) & (parameters['n2'] <= 2.0))
