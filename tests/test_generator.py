import os
import sys
from unittest.mock import patch

import pytest
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Generator
from cst import split_surface_t_values


def generator_config(shape_coefficient_count=10):
    return {
        'noise_dimension': 6,
        'cond_dim': 4,
        'gen_hid_node': 16,
        'gen_hid_layer': 2,
        'num_output_points': 100,
        'point_density_beta': 1.3,
        'gan_trailing_edge_crossing_point_count': 3,
        'gan_trailing_edge_crossing_te_weight': 80.0,
        'cst': {
            'shape_coefficient_count': shape_coefficient_count,
            'n1_range': [0.25, 1.0],
            'n2_range': [0.5, 2.0],
        },
    }


def coordinate_normalization():
    return {
        'x_min': torch.tensor(0.0),
        'x_max': torch.tensor(1.0),
        'y_min': torch.tensor(-0.2),
        'y_max': torch.tensor(0.2),
    }


@patch('model.torch.load', return_value=coordinate_normalization())
def test_generator_uses_cst_parameterization(_load):
    generator = Generator(generator_config())

    assert generator.parameter_dimension == 24

    parameters = torch.randn(3, generator.parameter_dimension, requires_grad=True)
    decoded_parameters = generator.decode_parameters(parameters)
    upper_coefficients, lower_coefficients, upper_te_y, lower_te_y, n1, n2 = (
        decoded_parameters
    )

    assert upper_coefficients.shape == (3, 10)
    assert lower_coefficients.shape == (3, 10)
    assert upper_te_y.shape == (3, 1)
    assert lower_te_y.shape == (3, 1)
    assert torch.all((n1 >= 0.25) & (n1 <= 1.0))
    assert torch.all((n2 >= 0.5) & (n2 <= 2.0))

    curve = generator.cst_layer(*decoded_parameters)
    assert curve.shape == (3, 100, 2)
    assert torch.equal(curve[:, 50, :], torch.zeros(3, 2))
    curve.square().mean().backward()
    assert torch.all(parameters.grad.abs().sum(dim=0) > 0)


@patch('model.torch.load', return_value=coordinate_normalization())
def test_generator_preserves_finite_trailing_edge_terms(_load):
    generator = Generator(generator_config())
    parameters = torch.zeros(1, generator.parameter_dimension)
    parameters[:, 20] = 0.03
    parameters[:, 21] = -0.02
    physical_curve = generator.cst_layer(*generator.decode_parameters(parameters))

    assert torch.equal(physical_curve[:, 0, 0], torch.ones(1))
    assert torch.equal(physical_curve[:, -1, 0], torch.ones(1))
    assert torch.equal(physical_curve[:, 50, :], torch.zeros(1, 2))
    assert torch.allclose(physical_curve[:, 0, 1], torch.tensor([0.03]))
    assert torch.allclose(physical_curve[:, -1, 1], torch.tensor([-0.02]))


@patch('model.torch.load', return_value=coordinate_normalization())
def test_trailing_edge_crossing_loss_matches_paper_regularizer(_load):
    generator = Generator(generator_config())
    parameters = torch.zeros(2, generator.parameter_dimension, requires_grad=True)
    with torch.no_grad():
        parameters[0, 20] = 0.01
        parameters[0, 21] = -0.02
        parameters[1, 20] = -0.01
        parameters[1, 21] = 0.02

    crossing_loss = generator.cst_layer.trailing_edge_crossing_loss(
        *generator.decode_parameters(parameters)
    )

    expected_weighted_x_sum = torch.sum(
        generator.cst_layer.trailing_edge_crossing_weights
        * generator.cst_layer.trailing_edge_crossing_x
    )
    expected_loss = 0.5 * 0.03 * expected_weighted_x_sum
    assert torch.allclose(crossing_loss, expected_loss, atol=1e-6)
    assert torch.allclose(
        generator.cst_layer.trailing_edge_crossing_weights,
        torch.tensor([80.0, 40.0, 0.0]),
    )
    assert torch.allclose(
        generator.cst_layer.trailing_edge_crossing_x,
        torch.tensor([1.0, 0.974078, 0.948315]),
        atol=1e-6,
    )
    crossing_loss.backward()
    assert torch.isfinite(parameters.grad).all()
    assert torch.allclose(parameters.grad[1, 20], -0.5 * expected_weighted_x_sum)
    assert torch.allclose(parameters.grad[1, 21], 0.5 * expected_weighted_x_sum)
    assert parameters.grad[0, 20] == 0.0
    assert parameters.grad[0, 21] == 0.0


@patch('model.torch.load', return_value=coordinate_normalization())
def test_generator_returns_trailing_edge_loss_without_changing_curve_output(_load):
    generator = Generator(generator_config())
    noise = torch.randn(3, generator.noise_dim)
    conditions = torch.randn(3, generator.cond_dim)

    expected_curve = generator(noise, conditions)
    actual_curve, crossing_loss = generator.generate_with_trailing_edge_crossing_loss(
        noise,
        conditions,
    )

    assert torch.equal(actual_curve, expected_curve)
    assert crossing_loss.ndim == 0
    assert crossing_loss >= 0.0


@patch('model.torch.load', return_value=coordinate_normalization())
def test_generator_rejects_insufficient_cst_shape_count(_load):
    config = generator_config(shape_coefficient_count=1)

    with pytest.raises(ValueError, match='must be at least 2'):
        Generator(config)


def test_split_surface_sampling_has_one_shared_leading_edge():
    upper_t, lower_t = split_surface_t_values(100, 1.3)

    assert upper_t.shape == (51,)
    assert lower_t.shape == (50,)
    assert upper_t[0] == 0.0
    assert upper_t[-1] == 1.0
    assert lower_t[0] == 0.0
    assert lower_t[-1] == 1.0
    assert torch.cat([upper_t, lower_t[1:]]).shape == (100,)
