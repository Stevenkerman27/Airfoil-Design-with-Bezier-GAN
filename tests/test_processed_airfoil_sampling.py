from pathlib import Path

import numpy as np
import yaml

from cst import split_surface_t_values
from utils import calculate_relative_thickness


def test_processed_airfoils_use_shared_split_surface_index():
    processed_dir = Path('foildata/processed_foil')
    paths = sorted(processed_dir.glob('*.dat'))
    assert paths
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    num_output_points = config['num_output_points']
    upper_t, _ = split_surface_t_values(
        num_output_points,
        config['point_density_beta'],
    )
    expected_le_index = upper_t.shape[0] - 1
    for path in paths:
        coordinates = np.loadtxt(path, skiprows=1)
        assert coordinates.shape == (num_output_points, 2), path
        assert int(np.argmin(coordinates[:, 0])) == expected_le_index, path
        assert (
            calculate_relative_thickness(coordinates)
            <= config['airfoil_preprocess']['max_relative_thickness']
        ), path
