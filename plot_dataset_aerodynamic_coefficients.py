import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml

from artifact_io import save_report_figure


PLOT_CONFIG_KEY = 'dataset_aerodynamic_coefficient_plot'
AXIS_PADDING_FRACTION = 0.05


def load_plot_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config[PLOT_CONFIG_KEY]


def load_coefficients(data_path):
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    if len(data) == 0:
        raise ValueError(f'Dataset is empty: {data_path}')

    alpha = np.array([item['y'][0].item() for item in data], dtype=np.float64)
    reynolds = np.array([item['y'][1].item() for item in data], dtype=np.float64)
    cl = np.array([item['y'][2].item() for item in data], dtype=np.float64)
    cm = np.array([item['y'][3].item() for item in data], dtype=np.float64)
    cd = np.array([item['cd'] for item in data], dtype=np.float64)
    return alpha, reynolds, cd, cl, cm


def padded_limits(values):
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        raise ValueError('Cannot derive plotting limits from constant values')
    padding = (upper - lower) * AXIS_PADDING_FRACTION
    return lower - padding, upper + padding


def condition_mask(alpha, reynolds, alpha_value, re_value):
    return np.isclose(alpha, alpha_value) & np.isclose(reynolds, re_value)


def plot_coefficients(plot_config):
    alpha, reynolds, cd, cl, cm = load_coefficients('model/airfoil_dataset.pt')
    alpha_values = plot_config['alpha_values']
    re_values = plot_config['re_values']

    selected_masks = {}
    for alpha_value in alpha_values:
        for re_value in re_values:
            mask = condition_mask(alpha, reynolds, alpha_value, re_value)
            if not np.any(mask):
                raise ValueError(
                    f'No samples for alpha={alpha_value} deg, Re={re_value:g}'
                )
            selected_masks[(alpha_value, re_value)] = mask

    combined_mask = np.logical_or.reduce(list(selected_masks.values()))
    cd_limits = Normalize(vmin=float(np.min(cd[combined_mask])), vmax=float(np.max(cd[combined_mask])))

    fig, axes = plt.subplots(
        len(alpha_values), len(re_values), figsize=(15, 12)
    )
    axes = np.asarray(axes).reshape(len(alpha_values), len(re_values))

    scatter = None
    for row, alpha_value in enumerate(alpha_values):
        for column, re_value in enumerate(re_values):
            axis = axes[row, column]
            mask = selected_masks[(alpha_value, re_value)]
            scatter = axis.scatter(
                cm[mask], cl[mask], c=cd[mask], cmap='viridis', norm=cd_limits,
                s=14, alpha=0.8, linewidths=0,
            )
            axis.set_title(f'alpha = {alpha_value:g} deg, Re = {re_value:,.0f}')
            axis.grid(True, linestyle='--', alpha=0.35)
            axis.set_xlim(padded_limits(cm[mask]))
            axis.set_ylim(padded_limits(cl[mask]))
            if row == len(alpha_values) - 1:
                axis.set_xlabel('Cm')
            if column == 0:
                axis.set_ylabel('Cl')

    fig.suptitle('Original Dataset Aerodynamic Coefficients', y=0.995)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.94, wspace=0.15, hspace=0.2)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Cd', pad=0.03)

    output_path = 'reports/dataset/original_aerodynamic_coefficients_3x3.png'
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    save_report_figure(fig, output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot original dataset aerodynamic coefficients for configured conditions'
    )
    parser.add_argument('--config', default='config.yaml', help='YAML configuration path')
    args = parser.parse_args()
    plot_coefficients(load_plot_config(args.config))


if __name__ == '__main__':
    main()
