import argparse
import copy
import optuna

from artifact_io import save_yaml
from train_surrogate import load_config, run_cross_validation


OPTUNA_CONFIG_KEY = 'optuna'
SEARCH_SPACE_KEY = 'search_space'
LOG_SCALE = 'log'
LINEAR_SCALE = 'linear'
OPTUNA_BEST_PARAMS_PATH = 'reports/surrogate/surrogate_optuna_best.yaml'


def build_pruner(optuna_config):
    pruner_name = optuna_config['pruner']
    if pruner_name == 'none':
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown optuna pruner: {pruner_name}")


def suggest_value(trial, name, spec):
    if not isinstance(spec, list):
        raise ValueError(f"Search space for {name} must be a list, got {type(spec)}")
    if len(spec) == 3 and spec[2] == LOG_SCALE:
        return trial.suggest_float(name, float(spec[0]), float(spec[1]), log=True)
    if len(spec) == 3 and spec[2] == LINEAR_SCALE:
        return trial.suggest_float(name, float(spec[0]), float(spec[1]))
    if len(spec) == 2 and all(isinstance(value, int) for value in spec):
        return trial.suggest_int(name, spec[0], spec[1])
    return trial.suggest_categorical(name, spec)


def apply_trial_params(config, trial):
    optuna_config = config[OPTUNA_CONFIG_KEY]
    search_space = optuna_config[SEARCH_SPACE_KEY]
    trial_config = copy.deepcopy(config)
    for name, spec in search_space.items():
        trial_config[name] = suggest_value(trial, name, spec)
    return trial_config


def save_best_result(path, study):
    result = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
    }
    save_yaml(path, result)


def run_optimization(config_path, n_trials_override=None):
    config = load_config(config_path)
    optuna_config = config[OPTUNA_CONFIG_KEY]
    n_trials = optuna_config['n_trials']
    if n_trials_override is not None:
        n_trials = n_trials_override

    study = optuna.create_study(
        study_name=optuna_config['study_name'],
        direction=optuna_config['direction'],
        pruner=build_pruner(optuna_config),
    )

    def objective(trial):
        trial_config = apply_trial_params(config, trial)
        metrics = run_cross_validation(trial_config, trial=trial)
        trial.set_user_attr('cv_loss_std', metrics['cv_loss_std'])
        trial.set_user_attr('fold_losses', metrics['fold_losses'])
        trial.set_user_attr('fold_mae', metrics['fold_mae'])
        trial.set_user_attr('fold_per_target_mae', metrics['fold_per_target_mae'])
        return metrics['cv_loss_mean']

    study.optimize(objective, n_trials=n_trials)
    save_best_result(OPTUNA_BEST_PARAMS_PATH, study)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best cross-validation loss: {study.best_value:.6f}")
    print(f'Saved best Optuna params to {OPTUNA_BEST_PARAMS_PATH}')
    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize aerodynamic surrogate hyperparameters with Optuna')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--n-trials', type=int, help='Override optuna.n_trials')
    args = parser.parse_args()
    run_optimization(
        args.config,
        n_trials_override=args.n_trials,
    )
