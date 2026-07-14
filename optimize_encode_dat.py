import argparse
import copy
import os
from pathlib import Path

import optuna
import yaml

from encode_dat import fit_bezier_airfoil_batch, load_coord_norm
from optimize_surrogate import build_pruner, suggest_value
from train_surrogate import load_config, resolve_device


OPTUNA_CONFIG_KEY = 'bezier_encode_optuna'
SEARCH_SPACE_KEY = 'search_space'


def list_airfoil_paths(data_dir):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Airfoil data directory not found: {data_dir}")
    paths = sorted(root.glob('*.dat'))
    if len(paths) == 0:
        raise ValueError(f"No .dat files found in {data_dir}")
    return [str(path) for path in paths]


def set_nested_value(config, dotted_name, value):
    parts = dotted_name.split('.')
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def apply_trial_params(config, trial):
    optuna_config = config[OPTUNA_CONFIG_KEY]
    search_space = optuna_config[SEARCH_SPACE_KEY]
    trial_config = copy.deepcopy(config)
    for name, spec in search_space.items():
        set_nested_value(trial_config, name, suggest_value(trial, name, spec))
    return trial_config


def validate_encode_search_space(config):
    curve_mode = config['bezier_encode']['curve_mode']
    search_space = config[OPTUNA_CONFIG_KEY][SEARCH_SPACE_KEY]
    has_total_control_points = 'num_control_points' in search_space
    has_surface_control_points = 'bezier_encode.surface_control_points' in search_space
    if curve_mode == 'single':
        if not has_total_control_points:
            raise ValueError(
                "single curve mode requires num_control_points in bezier_encode_optuna.search_space"
            )
        if has_surface_control_points:
            raise ValueError(
                "single curve mode must not search bezier_encode.surface_control_points"
            )
    elif curve_mode == 'split_surface':
        if not has_surface_control_points:
            raise ValueError(
                "split_surface mode requires bezier_encode.surface_control_points "
                "in bezier_encode_optuna.search_space"
            )
        if has_total_control_points:
            raise ValueError(
                "split_surface mode must not search top-level num_control_points"
            )
    else:
        raise ValueError(f"Unsupported bezier_encode.curve_mode: {curve_mode}")


def summarize_results(results):
    mae_values = []
    mse_values = []
    le_mae_values = []
    le_mse_values = []
    max_point_errors = []
    surface_control_point_counts = None
    total_control_points = None
    for item in results:
        mae_values.extend(item['mae_per_airfoil'].tolist())
        mse_values.extend(item['mse_per_airfoil'].tolist())
        le_mae_values.extend(item['leading_edge_mae_per_airfoil'].tolist())
        le_mse_values.extend(item['leading_edge_mse_per_airfoil'].tolist())
        max_point_errors.extend(item['max_point_error_per_airfoil'].tolist())
        if 'surface_control_point_counts' in item:
            surface_control_point_counts = item['surface_control_point_counts']
        if 'total_control_points' in item:
            total_control_points = item['total_control_points']
    metrics = {
        'mean_mae': sum(mae_values) / len(mae_values),
        'mean_mse': sum(mse_values) / len(mse_values),
        'mean_leading_edge_mae': sum(le_mae_values) / len(le_mae_values),
        'mean_leading_edge_mse': sum(le_mse_values) / len(le_mse_values),
        'max_point_error': max(max_point_errors),
    }
    if surface_control_point_counts is not None:
        metrics['surface_control_point_counts'] = surface_control_point_counts
    if total_control_points is not None:
        metrics['total_control_points'] = total_control_points
    return metrics


def save_best_result(path, study):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'best_metrics': study.best_trial.user_attrs,
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)


def select_airfoil_paths(paths, max_airfoils):
    if max_airfoils is None:
        return paths
    if max_airfoils <= 0:
        raise ValueError(f"max_airfoils must be positive when set, got {max_airfoils}")
    return paths[:max_airfoils]


def evaluate_trial(trial_config, airfoil_paths, coord_norm, device):
    results = []
    batch_size = trial_config['bezier_encode']['batch_size']
    if batch_size <= 0:
        raise ValueError(f"bezier_encode.batch_size must be positive, got {batch_size}")
    total_airfoils = len(airfoil_paths)
    for start in range(0, total_airfoils, batch_size):
        batch_paths = airfoil_paths[start:start + batch_size]
        result = fit_bezier_airfoil_batch(
            batch_paths,
            trial_config,
            coord_norm,
            device,
            verbose=False,
        )
        results.append(result)
        encoded_count = min(start + batch_size, total_airfoils)
        if (
            encoded_count % trial_config[OPTUNA_CONFIG_KEY]['log_every_airfoils'] == 0
            or encoded_count == total_airfoils
        ):
            print(f"Encoded {encoded_count}/{total_airfoils} airfoils for current trial")
    return summarize_results(results)


def run_optimization(config_path, n_trials_override=None, max_airfoils_override=None):
    config = load_config(config_path)
    validate_encode_search_space(config)
    optuna_config = config[OPTUNA_CONFIG_KEY]
    n_trials = optuna_config['n_trials']
    if n_trials_override is not None:
        n_trials = n_trials_override

    airfoil_paths = list_airfoil_paths(optuna_config['data_dir'])
    max_airfoils = optuna_config['max_airfoils']
    if max_airfoils_override is not None:
        max_airfoils = max_airfoils_override
    airfoil_paths = select_airfoil_paths(airfoil_paths, max_airfoils)

    coord_norm = load_coord_norm(config['bezier_encode']['coord_norm_path'])
    device = resolve_device(config)

    print(f"Optimizing Bezier encoding on {len(airfoil_paths)} airfoils")
    study = optuna.create_study(
        study_name=optuna_config['study_name'],
        direction=optuna_config['direction'],
        pruner=build_pruner(optuna_config),
    )

    def objective(trial):
        trial_config = apply_trial_params(config, trial)
        metrics = evaluate_trial(trial_config, airfoil_paths, coord_norm, device)
        total_control_points = metrics['total_control_points']
        control_point_penalty_weight = float(optuna_config['control_point_penalty_weight'])
        objective_value = (
            metrics['mean_mae']
            + control_point_penalty_weight * total_control_points
        )
        trial.set_user_attr('mean_mae', metrics['mean_mae'])
        trial.set_user_attr('mean_mse', metrics['mean_mse'])
        trial.set_user_attr('mean_leading_edge_mae', metrics['mean_leading_edge_mae'])
        trial.set_user_attr('mean_leading_edge_mse', metrics['mean_leading_edge_mse'])
        trial.set_user_attr('max_point_error', metrics['max_point_error'])
        trial.set_user_attr('total_control_points', total_control_points)
        trial.set_user_attr(
            'surface_control_points',
            trial_config['bezier_encode']['surface_control_points'],
        )
        if 'surface_control_point_counts' in metrics:
            trial.set_user_attr(
                'surface_control_point_counts',
                metrics['surface_control_point_counts'],
            )
        return objective_value

    study.optimize(objective, n_trials=n_trials)
    save_best_result(optuna_config['best_params_path'], study)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best objective: {study.best_value:.8f}")
    print(f"Best mean MAE: {study.best_trial.user_attrs['mean_mae']:.8f}")
    print(f"Best mean MSE: {study.best_trial.user_attrs['mean_mse']:.8f}")
    print(f"Best total_control_points: {study.best_trial.user_attrs['total_control_points']}")
    print(f"Best surface_control_points: {study.best_trial.user_attrs['surface_control_points']}")
    print(f"Saved best Optuna params to {optuna_config['best_params_path']}")
    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Bezier airfoil encoding hyperparameters with Optuna')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--n-trials', type=int, help='Override bezier_encode_optuna.n_trials')
    parser.add_argument('--max-airfoils', type=int, help='Override bezier_encode_optuna.max_airfoils')
    args = parser.parse_args()
    run_optimization(
        args.config,
        n_trials_override=args.n_trials,
        max_airfoils_override=args.max_airfoils,
    )
