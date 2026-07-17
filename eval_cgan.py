import argparse
import concurrent.futures
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from foildata.xfoil import run_xfoil_single
from model import Generator
from surrogate_split import load_split_indices, resolve_surrogate_dataset_config
from utils import calculate_relative_thickness


DEFAULT_N_COND = 60
DEFAULT_K_SAMPLES = 20
DEFAULT_TOP_M = 5
GAN_LABEL_ORDER = ['alpha', 'Re', 'CL', 'CM']


def resolve_device(config):
    device_cfg = config['device'].lower()
    if device_cfg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_cfg == 'cpu':
        return torch.device('cpu')
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(f"Unknown device configuration: {config['device']}")


def load_eval_conditions(config, split_name, count):
    _, dataset_config = resolve_surrogate_dataset_config(config)
    raw_data = torch.load(dataset_config['data_path'], map_location='cpu', weights_only=True)
    _, split_indices = load_split_indices(raw_data, config)
    indices = split_indices[split_name]
    if count > len(indices):
        raise ValueError(
            f'Requested {count} evaluation conditions from {split_name}, only {len(indices)} available'
        )

    generator = torch.Generator().manual_seed(config['surrogate_seed'])
    selected = torch.randperm(len(indices), generator=generator)[:count].tolist()
    conditions = torch.stack([raw_data[indices[index]]['y'] for index in selected]).float()
    if conditions.size(1) != len(GAN_LABEL_ORDER):
        raise ValueError(
            f'Expected {len(GAN_LABEL_ORDER)} GAN labels, got {conditions.size(1)}'
        )
    return conditions.numpy()


def plot_heatmap(alpha, reynolds, errors, title, path):
    plt.figure(figsize=(9, 7))
    try:
        contour = plt.tricontourf(alpha, reynolds, errors, levels=20, cmap='jet')
        plt.colorbar(contour, label='Mean absolute error + variance penalty')
    except Exception as exc:
        print(f'Warning: could not create contour plot for {title}: {exc}')
        scatter = plt.scatter(alpha, reynolds, c=errors, cmap='jet', edgecolors='k', alpha=0.8)
        plt.colorbar(scatter, label='Mean absolute error + variance penalty')
    plt.scatter(alpha, reynolds, c='k', s=10, alpha=0.4)
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Reynolds Number')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate_xfoil(task):
    coords, reynolds, alpha, target_cl, target_cm, cm_weight, cl_weight = task
    thickness = calculate_relative_thickness(coords)
    xfoil_result = run_xfoil_single(coords, reynolds, alpha, return_all=True)
    if xfoil_result is None:
        return {
            'coords': coords,
            'alpha': alpha,
            're': reynolds,
            'target_cl': target_cl,
            'target_cm': target_cm,
            'thickness': thickness,
            'cl': np.nan,
            'cm': np.nan,
            'cd': np.nan,
            'cl_err': np.nan,
            'cm_err': np.nan,
            'weighted_err': np.nan,
            'success': False,
        }

    cl = xfoil_result['CL']
    cm = xfoil_result['CM']
    cd = xfoil_result['CD']
    cl_err = abs(cl - target_cl)
    cm_err = abs(cm - target_cm)
    return {
        'coords': coords,
        'alpha': alpha,
        're': reynolds,
        'target_cl': target_cl,
        'target_cm': target_cm,
        'thickness': thickness,
        'cl': cl,
        'cm': cm,
        'cd': cd,
        'cl_err': cl_err,
        'cm_err': cm_err,
        'weighted_err': cm_weight * cm_err + cl_weight * cl_err,
        'success': True,
    }


def load_generator(model_path, config, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model {model_path} not found')
    generator = Generator(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or 'generator_state_dict' not in checkpoint:
        raise ValueError(f'Model {model_path} is not a GAN checkpoint')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator


def evaluate_model(model_path, tag, config, device, split_name, n_cond, k_samples, top_m):
    print(f'--- Evaluating {tag} on GAN {split_name} conditions: {model_path} ---')
    generator = load_generator(model_path, config, device)
    norm_params = torch.load('model/cond_norm.pt', map_location=device, weights_only=True)
    coord_norm = torch.load('model/coord_norm.pt', map_location=device, weights_only=True)
    cond_mean = norm_params['mean'].to(device)
    cond_std = norm_params['std'].to(device)
    conditions = load_eval_conditions(config, split_name, n_cond)
    cm_weight, cl_weight = config['gan_surrogate_target_loss_weights']

    expanded_conditions = np.repeat(conditions, k_samples, axis=0)
    condition_tensor = torch.tensor(expanded_conditions, dtype=torch.float32, device=device)
    normalized_conditions = (condition_tensor - cond_mean) / cond_std
    noise = torch.randn(len(expanded_conditions), config['noise_dimension'], device=device)
    with torch.no_grad():
        coords = generator(noise, normalized_conditions)

    point_count = config['num_output_points']
    coords = coords.view(len(expanded_conditions), point_count, 2)
    coords[:, :, 0] = coords[:, :, 0] * (coord_norm['x_max'].to(device) - coord_norm['x_min'].to(device) + 1e-8) + coord_norm['x_min'].to(device)
    coords[:, :, 1] = coords[:, :, 1] * (coord_norm['y_max'].to(device) - coord_norm['y_min'].to(device) + 1e-8) + coord_norm['y_min'].to(device)
    generated_coords = coords.cpu().numpy()

    tasks = []
    for index, condition in enumerate(expanded_conditions):
        alpha, reynolds, target_cl, target_cm = condition
        tasks.append((generated_coords[index], reynolds, alpha, target_cl, target_cm, cm_weight, cl_weight))
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(executor.map(evaluate_xfoil, tasks))

    results_by_condition = [results[index * k_samples:(index + 1) * k_samples] for index in range(n_cond)]
    alpha_values, reynolds_values, cm_scores, cl_scores, weighted_scores = [], [], [], [], []
    variance_weight = config['eval_var_weight']
    for condition_results in results_by_condition:
        valid = [result for result in condition_results if result['success']]
        if not valid:
            continue
        alpha_values.append(valid[0]['alpha'])
        reynolds_values.append(valid[0]['re'])
        cm_errors = np.array([result['cm_err'] for result in valid])
        cl_errors = np.array([result['cl_err'] for result in valid])
        weighted_errors = np.array([result['weighted_err'] for result in valid])
        cm_scores.append(np.mean(cm_errors) + variance_weight * np.var(cm_errors))
        cl_scores.append(np.mean(cl_errors) + variance_weight * np.var(cl_errors))
        weighted_scores.append(np.mean(weighted_errors) + variance_weight * np.var(weighted_errors))

    os.makedirs('model', exist_ok=True)
    plot_heatmap(alpha_values, reynolds_values, cm_scores, f'{tag} CM error', f'model/eval_{tag.lower()}_cm.png')
    plot_heatmap(alpha_values, reynolds_values, cl_scores, f'{tag} CL error', f'model/eval_{tag.lower()}_cl.png')
    plot_heatmap(alpha_values, reynolds_values, weighted_scores, f'{tag} weighted CM/CL error', f'model/eval_{tag.lower()}_weighted.png')

    valid_results = [result for result in results if result['success']]
    valid_results.sort(key=lambda result: result['weighted_err'])
    os.makedirs('foildata/gen', exist_ok=True)
    for index, result in enumerate(valid_results[:top_m]):
        filename = (
            f"{tag}_Top{index + 1}_Werr{result['weighted_err']:.5f}_"
            f"Cl{result['cl']:.4f}_Cm{result['cm']:.4f}_Cd{result['cd']:.5f}.dat"
        )
        path = os.path.join('foildata/gen', filename)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(
                f"{tag}_Werr_{result['weighted_err']:.5f}_CL_{result['cl']:.4f}_"
                f"CM_{result['cm']:.4f}_Thickness_{result['thickness']:.4f}\n"
            )
            for point in result['coords']:
                file.write(f'{point[0]:.6f} {point[1]:.6f}\n')

    if not valid_results:
        raise ValueError('XFoil did not converge for any generated evaluation airfoil')
    print(f"Valid XFoil samples: {len(valid_results)}/{len(results)}")
    print(f"CM MAE: {np.mean([result['cm_err'] for result in valid_results]):.6f}")
    print(f"CL MAE: {np.mean([result['cl_err'] for result in valid_results]):.6f}")
    print(f"Weighted CM/CL MAE: {np.mean([result['weighted_err'] for result in valid_results]):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a 4D-condition CWGAN-GP model with XFoil')
    parser.add_argument('--model', default='model/gan_final.pt', help='GAN checkpoint path')
    parser.add_argument('--tag', default='GAN', help='Output tag')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Condition split')
    parser.add_argument('--n-cond', type=int, default=DEFAULT_N_COND, help='Number of held-out conditions')
    parser.add_argument('--k-samples', type=int, default=DEFAULT_K_SAMPLES, help='Generations per condition')
    parser.add_argument('--top-m', type=int, default=DEFAULT_TOP_M, help='Saved top generated airfoils')
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    evaluate_model(
        args.model,
        args.tag,
        config,
        resolve_device(config),
        args.split,
        args.n_cond,
        args.k_samples,
        args.top_m,
    )


if __name__ == '__main__':
    main()
