import argparse
import copy
import os

import optuna
import yaml

from train_surrogate import load_config, train_surrogate_from_config


OPTUNA_CONFIG_KEY = 'optuna'
SEARCH_SPACE_KEY = 'search_space'
LOG_SCALE = 'log'
LINEAR_SCALE = 'linear'


def build_pruner(optuna_config):
    pruner_name = optuna_config['pruner']
    warmup_epochs = optuna_config['warmup_epochs']
    if pruner_name == 'median':
        return optuna.pruners.MedianPruner(n_warmup_steps=warmup_epochs)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)


def run_optimization(config_path, n_trials_override=None, trial_epochs_override=None):
    config = load_config(config_path)
    optuna_config = config[OPTUNA_CONFIG_KEY]
    n_trials = optuna_config['n_trials']
    trial_epochs = optuna_config['trial_epochs']
    if n_trials_override is not None:
        n_trials = n_trials_override
    if trial_epochs_override is not None:
        trial_epochs = trial_epochs_override

    study = optuna.create_study(
        study_name=optuna_config['study_name'],
        direction=optuna_config['direction'],
        pruner=build_pruner(optuna_config),
    )

    def objective(trial):
        trial_config = apply_trial_params(config, trial)
        metrics = train_surrogate_from_config(
            trial_config,
            epochs_override=trial_epochs,
            save_artifacts=False,
            trial=trial,
        )
        trial.set_user_attr('best_epoch', metrics['best_epoch'])
        trial.set_user_attr('best_val_mae', metrics['best_val_mae'])
        return metrics['best_val_loss']

    study.optimize(objective, n_trials=n_trials)
    save_best_result(optuna_config['best_params_path'], study)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"Saved best Optuna params to {optuna_config['best_params_path']}")
    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize aerodynamic surrogate hyperparameters with Optuna')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--n-trials', type=int, help='Override optuna.n_trials')
    parser.add_argument('--trial-epochs', type=int, help='Override optuna.trial_epochs')
    args = parser.parse_args()
    run_optimization(
        args.config,
        n_trials_override=args.n_trials,
        trial_epochs_override=args.trial_epochs,
    )
