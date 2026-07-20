import numpy as np
import torch

from foildata.manage_foildata import normalize_airfoil_chord_coordinates
from train import normalize_surrogate_coords


def test_normalize_airfoil_chord_coordinates_uses_local_chord_frame():
    coordinates = np.array([
        [5.0, 3.0],
        [4.0, 2.5],
        [3.0, 2.0],
        [4.0, 1.5],
        [5.0, 1.0],
    ])

    normalized = normalize_airfoil_chord_coordinates(coordinates, 2)

    np.testing.assert_allclose(normalized[2], [0.0, 0.0])
    np.testing.assert_allclose(normalized[0], [1.0, 0.5])
    np.testing.assert_allclose(normalized[-1], [1.0, -0.5])


def test_batched_chord_normalization_is_pose_invariant_and_differentiable():
    base_coordinates = torch.tensor([
        [1.0, 0.1],
        [0.5, 0.2],
        [0.0, 0.0],
        [0.5, -0.2],
        [1.0, -0.1],
    ])
    angle = torch.tensor(0.4)
    rotation = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle)]),
        torch.stack([torch.sin(angle), torch.cos(angle)]),
    ])
    transformed_coordinates = 2.5 * (base_coordinates @ rotation.T) + torch.tensor([3.0, -4.0])
    coordinates = torch.stack([base_coordinates, transformed_coordinates]).requires_grad_()

    normalized = normalize_airfoil_chord_coordinates(coordinates)

    torch.testing.assert_close(normalized[0], base_coordinates)
    torch.testing.assert_close(normalized[1], base_coordinates)
    normalized[:, 1:-1].square().sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert torch.any(coordinates.grad != 0.0)


def test_surrogate_coordinate_normalization_uses_local_chord_frame_first():
    physical_coordinates = torch.tensor([[
        [5.0, 3.0],
        [4.0, 2.5],
        [3.0, 2.0],
        [4.0, 1.5],
        [5.0, 1.0],
    ]])
    coord_stats = {
        'x_min': torch.tensor(0.0),
        'x_max': torch.tensor(1.0),
        'y_min': torch.tensor(-0.5),
        'y_max': torch.tensor(0.5),
    }
    expected = torch.tensor([[
        [1.0, 1.0],
        [0.5, 0.75],
        [0.0, 0.5],
        [0.5, 0.25],
        [1.0, 0.0],
    ]])

    normalized = normalize_surrogate_coords(physical_coordinates, coord_stats)

    torch.testing.assert_close(normalized, expected.view(1, -1))
