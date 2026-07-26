import argparse
import concurrent.futures
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from artifact_io import save_report_figure, save_yaml
from generated_surrogate_utils import (
    GeneratedSurrogateDataset,
    evaluate_generated_surrogate_metrics,
)
from generated_xfoil_utils import (
    prepare_generated_xfoil_record,
    run_xfoil_for_generated_record,
)
from gan_conditions import GAN_LABEL_ORDER
from gan_exploration import (
    build_condition_record_strata,
    build_condition_statistics,
    sample_condition_from_stratum,
)
from model import AerodynamicSurrogate, Discriminator, Generator
from surrogate_split import load_cross_validation_manifest
from train import (
    build_surrogate_conditions,
    compute_generator_auxiliary_losses,
    compute_gradient_penalty,
    denormalize_gan_coords,
    load_gan_auxiliary_stats,
    normalize_gan_coords,
    save_checkpoint,
)
from train_surrogate import (
    AirfoilSurrogateDataset,
    build_weighted_mse_loss,
    load_config,
    resolve_device,
    set_training_seed,
)


ALTERNATING_CONFIG_KEY = 'alternating_finetuning'
ALTERNATING_DATASET_VERSION = 1
ALTERNATING_CHECKPOINT_POLICY = 'alternating_fixed_epoch'
ALTERNATING_CONFIG_KEYS = (
    'successful_samples_per_operating_condition',
    'max_xfoil_attempts_per_operating_condition',
    'noise_samples_per_condition',
    'generation_batch_size',
    'xfoil_timeout_seconds',
    'checkpoint_interval_collections',
    'existing_to_sampled_ratios',
    'mahalanobis_radii',
    'surrogate_epochs_per_round',
    'surrogate_learning_rate',
    'surrogate_generated_batch_size',
    'surrogate_original_replay_batch_size',
    'surrogate_historical_to_new_ratio',
    'surrogate_lambda_generated',
    'surrogate_lambda_original',
    'surrogate_lambda_anchor',
    'gan_epochs_per_round',
    'gan_learning_rate',
    'gan_batch_size',
    'gan_real_original_to_generated_ratio',
    'gan_adversarial_weight',
    'gan_aerodynamic_weight',
)
SURROGATE_TARGET_ORDER = ['CM', 'CL', 'CD']
SURROGATE_DATASET_PATH = 'model/airfoil_dataset.pt'
SURROGATE_NORM_PATH = 'model/surrogate_airfoil_group_norm.pt'
INITIAL_GAN_CHECKPOINT_PATH = 'model/gan_final.pt'
INITIAL_SURROGATE_CHECKPOINT_PATH = 'model/surrogate_airfoil_group_best.pt'
ACCUMULATED_DATASET_PATH = 'model/alternating_generated_dataset.pt'
ALTERNATING_CHECKPOINT_DIRECTORY = 'model/alternating_checkpoints'
ALTERNATING_REPORT_DIRECTORY = 'reports/alternating'
ALTERNATING_PLOT_DIRECTORY = 'reports/alternating'
ALTERNATING_RESET_ARTIFACTS = (
    ('accumulated dataset', ACCUMULATED_DATASET_PATH, False),
    ('checkpoint directory', ALTERNATING_CHECKPOINT_DIRECTORY, True),
    ('report directory', ALTERNATING_REPORT_DIRECTORY, True),
)


def _require_parent_directory(path, name):
    if not os.path.dirname(path):
        raise ValueError(f'{name} must include a parent directory: {path}')


def _validate_ratio(value, name, count=None):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name} must be a two-item list')
    if any(not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f'{name} must contain non-negative integers')
    if sum(value) == 0:
        raise ValueError(f'{name} must contain at least one positive value')
    if count is not None and len(value) != count:
        raise ValueError(f'{name} must contain {count} items')


def resolve_alternating_config(config):
    if ALTERNATING_CONFIG_KEY not in config:
        raise ValueError(f'config is missing {ALTERNATING_CONFIG_KEY}')
    values = config[ALTERNATING_CONFIG_KEY]
    missing = [key for key in ALTERNATING_CONFIG_KEYS if key not in values]
    if missing:
        raise ValueError(f'{ALTERNATING_CONFIG_KEY} is missing required keys: {missing}')
    integer_keys = (
        'successful_samples_per_operating_condition',
        'max_xfoil_attempts_per_operating_condition',
        'noise_samples_per_condition',
        'generation_batch_size',
        'xfoil_timeout_seconds',
        'checkpoint_interval_collections',
        'surrogate_epochs_per_round',
        'surrogate_generated_batch_size',
        'surrogate_original_replay_batch_size',
        'gan_epochs_per_round',
        'gan_batch_size',
    )
    for key in integer_keys:
        if not isinstance(values[key], int) or values[key] <= 0:
            raise ValueError(f'{ALTERNATING_CONFIG_KEY}.{key} must be a positive integer')
    float_keys = (
        'surrogate_learning_rate',
        'gan_learning_rate',
        'gan_adversarial_weight',
        'gan_aerodynamic_weight',
    )
    for key in float_keys:
        if float(values[key]) <= 0.0:
            raise ValueError(f'{ALTERNATING_CONFIG_KEY}.{key} must be positive')
    for key in (
        'surrogate_lambda_generated',
        'surrogate_lambda_original',
        'surrogate_lambda_anchor',
    ):
        if float(values[key]) < 0.0:
            raise ValueError(f'{ALTERNATING_CONFIG_KEY}.{key} must be non-negative')
    if (
        float(values['surrogate_lambda_generated']) == 0.0
        and float(values['surrogate_lambda_original']) == 0.0
    ):
        raise ValueError('At least one alternating surrogate data loss weight must be positive')
    ratios = values['existing_to_sampled_ratios']
    radii = values['mahalanobis_radii']
    if not isinstance(ratios, list) or not ratios:
        raise ValueError('existing_to_sampled_ratios must be a non-empty list')
    if not isinstance(radii, list) or len(radii) != len(ratios):
        raise ValueError('mahalanobis_radii must match existing_to_sampled_ratios length')
    for index, ratio in enumerate(ratios):
        _validate_ratio(ratio, f'existing_to_sampled_ratios[{index}]')
    for index, radius in enumerate(radii):
        if float(radius) <= 0.0:
            raise ValueError(f'mahalanobis_radii[{index}] must be positive')
    _validate_ratio(values['surrogate_historical_to_new_ratio'], 'surrogate_historical_to_new_ratio')
    _validate_ratio(values['gan_real_original_to_generated_ratio'], 'gan_real_original_to_generated_ratio')
    return values


def _round_name(round_index):
    return f'round_{round_index + 1:03d}'


def _round_path(directory, round_index, suffix):
    return os.path.join(directory, f'{_round_name(round_index)}_{suffix}')


def load_accumulated_dataset(path):
    if not os.path.exists(path):
        return {
            'version': ALTERNATING_DATASET_VERSION,
            'records': [],
            'completed_round_indices': [],
        }
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['version'] != ALTERNATING_DATASET_VERSION:
        raise ValueError(f'Unexpected alternating dataset version: {state["version"]}')
    if not isinstance(state['records'], list):
        raise ValueError('Alternating dataset records must be a list')
    if not isinstance(state['completed_round_indices'], list):
        raise ValueError('Alternating completed_round_indices must be a list')
    return state


def save_accumulated_dataset(path, state):
    _require_parent_directory(path, 'Alternating accumulated dataset path')
    torch.save(state, path)


def _actual_condition(record):
    if record['status'] != 'success':
        raise ValueError('Actual condition is only defined for successful XFoil records')
    if 'actual_condition' not in record:
        raise ValueError('Successful alternating record is missing actual_condition')
    condition = record['actual_condition'].float()
    if condition.ndim != 1 or condition.numel() != len(GAN_LABEL_ORDER):
        raise ValueError('actual_condition must contain [alpha, Re, CL, CM]')
    if not bool(torch.isfinite(condition).all().item()):
        raise ValueError('actual_condition must be finite')
    return condition


def successful_records(records):
    result = []
    for record in records:
        if record['status'] != 'success':
            continue
        required = ('coords', 'condition', 'targets', 'cache_key', 'round_index')
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f'Alternating success record is missing keys: {missing}')
        _actual_condition(record)
        targets = record['targets'].float()
        if targets.numel() != len(SURROGATE_TARGET_ORDER):
            raise ValueError('Alternating success targets must contain [CM, CL, CD]')
        if not bool(torch.isfinite(targets).all().item()):
            raise ValueError('Alternating success targets must be finite')
        result.append(record)
    return result


def build_known_condition_records(raw_data, development_indices, generated_records):
    records = []
    for dataset_index in development_indices:
        records.append({
            'record_id': f'original:{dataset_index}',
            'condition': raw_data[dataset_index]['y'].float(),
        })
    for record in successful_records(generated_records):
        records.append({
            'record_id': f'generated:{record["cache_key"]}',
            'condition': _actual_condition(record),
        })
    return records


def _source_for_anchor(anchor_index, ratio):
    cycle_position = anchor_index % sum(ratio)
    return 'empirical' if cycle_position < ratio[0] else 'covariance'


def _stratum_seed(config, round_index, stratum_index, anchor_index, noise_index):
    return (
        config['surrogate_seed']
        + 1000003 * (round_index + 1)
        + 10007 * (stratum_index + 1)
        + 101 * (anchor_index + 1)
        + noise_index
    )


def build_stratum_candidate_requests(
    config,
    alternating_config,
    round_index,
    stratum_index,
    stratum_key,
    stratum_records,
    anchor_index,
    remaining_attempts,
):
    if remaining_attempts <= 0:
        return []
    ratio = alternating_config['existing_to_sampled_ratios'][round_index]
    source = _source_for_anchor(anchor_index, ratio)
    condition_generator = torch.Generator().manual_seed(
        _stratum_seed(config, round_index, stratum_index, anchor_index, 0)
    )
    sampled = sample_condition_from_stratum(
        stratum_records,
        source,
        float(alternating_config['mahalanobis_radii'][round_index]),
        condition_generator,
    )
    candidate_count = min(
        alternating_config['noise_samples_per_condition'], remaining_attempts
    )
    requests = []
    round_id = _round_name(round_index)
    stratum_name = f'{stratum_key[0]:g}_{stratum_key[1]:g}'
    for noise_index in range(candidate_count):
        noise_generator = torch.Generator().manual_seed(
            _stratum_seed(
                config, round_index, stratum_index, anchor_index, noise_index + 1
            )
        )
        requests.append({
            'request_id': f'{round_id}:{stratum_name}:{anchor_index}:{noise_index}',
            'source_dataset_index': str(sampled['source_record_id']),
            'noise_index': noise_index,
            'condition': sampled['condition'].clone(),
            'noise': torch.randn(config['noise_dimension'], generator=noise_generator),
            'source': sampled['source'],
            'mahalanobis_radius': sampled['mahalanobis_radius'],
            'stratum_key': stratum_key,
            'anchor_index': anchor_index,
        })
    return requests, sampled


def _set_actual_condition(record):
    if record['status'] != 'success':
        return
    requested = record['condition'].float()
    targets = record['targets'].float()
    actual = requested.clone()
    actual[2] = targets[1]
    actual[3] = targets[0]
    record['actual_condition'] = actual


def _copy_cached_xfoil_result(record, cached_record):
    if cached_record['status'] in ('success', 'quota_excess_success'):
        record['status'] = 'success'
        record['targets'] = cached_record['targets'].clone()
        _set_actual_condition(record)
    else:
        record['status'] = cached_record['status']
    if record['status'] != 'success' and 'failure_reason' in cached_record:
        record['failure_reason'] = cached_record['failure_reason']


class RoundRecordCollector:
    def __init__(
        self,
        config,
        alternating_config,
        state,
        generator_checkpoint_path,
        round_index,
        device,
    ):
        self.config = config
        self.alternating_config = alternating_config
        self.state = state
        self.round_index = round_index
        self.device = device
        self.current_records = {
            record['request_id']: record
            for record in state['records']
            if record.get('round_index') == round_index
        }
        self.cached_by_key = {
            record['cache_key']: record
            for record in state['records']
            if 'cache_key' in record
            and record['status'] in ('success', 'quota_excess_success', 'xfoil_failed')
        }
        checkpoint = torch.load(generator_checkpoint_path, map_location=device, weights_only=True)
        if 'generator_state_dict' not in checkpoint:
            raise ValueError(
                f'GAN checkpoint is missing generator_state_dict: {generator_checkpoint_path}'
            )
        self.generator = Generator(config).to(device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        self.auxiliary_stats = load_gan_auxiliary_stats(config, device)

    def collect(self, requests):
        if not requests:
            return []
        for start in range(0, len(requests), self.alternating_config['generation_batch_size']):
            request_batch = requests[start:start + self.alternating_config['generation_batch_size']]
            missing_requests = [
                request for request in request_batch
                if request['request_id'] not in self.current_records
            ]
            if not missing_requests:
                continue
            conditions = torch.stack([
                request['condition'] for request in missing_requests
            ]).to(self.device)
            noise = torch.stack([request['noise'] for request in missing_requests]).to(self.device)
            normalized_conditions = (
                conditions - self.auxiliary_stats['gan_cond_mean']
            ) / self.auxiliary_stats['gan_cond_std']
            with torch.no_grad():
                normalized_coords = self.generator(noise, normalized_conditions)
                physical_coords = denormalize_gan_coords(
                    normalized_coords,
                    self.auxiliary_stats['gan_coord'],
                    self.config['num_output_points'],
                ).cpu()
            prepared_records = []
            for index, request in enumerate(missing_requests):
                record = prepare_generated_xfoil_record(
                    request, physical_coords[index], _round_name(self.round_index)
                )
                record['round_index'] = self.round_index
                record['requested_condition'] = request['condition'].clone()
                record['condition_source'] = request['source']
                record['mahalanobis_radius'] = request['mahalanobis_radius']
                record['stratum_key'] = request['stratum_key']
                record['anchor_index'] = request['anchor_index']
                prepared_records.append(record)
            pending_by_key = {}
            for record in prepared_records:
                if record['status'] != 'pending':
                    continue
                cached = self.cached_by_key.get(record['cache_key'])
                if cached is not None:
                    _copy_cached_xfoil_result(record, cached)
                    continue
                pending_by_key.setdefault(record['cache_key'], record)
            pending_records = list(pending_by_key.values())
            if pending_records:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config['max_workers']
                ) as executor:
                    completed_records = list(executor.map(
                        lambda item: run_xfoil_for_generated_record(
                            item,
                            self.alternating_config['xfoil_timeout_seconds'],
                        ),
                        pending_records,
                    ))
                completed_by_key = {
                    record['cache_key']: record for record in completed_records
                }
                for record in prepared_records:
                    if record['status'] != 'pending':
                        continue
                    _copy_cached_xfoil_result(
                        record, completed_by_key[record['cache_key']]
                    )
            for record in prepared_records:
                _set_actual_condition(record)
                self.state['records'].append(record)
                self.current_records[record['request_id']] = record
                if 'cache_key' in record and record['status'] in (
                    'success', 'quota_excess_success', 'xfoil_failed'
                ):
                    self.cached_by_key[record['cache_key']] = record
        return [self.current_records[request['request_id']] for request in requests]


def collect_balanced_round_records(
    config,
    alternating_config,
    state,
    generator_checkpoint_path,
    round_index,
    known_condition_records,
    device,
):
    strata = build_condition_record_strata(known_condition_records)
    target_success_count = alternating_config['successful_samples_per_operating_condition']
    max_attempt_count = alternating_config['max_xfoil_attempts_per_operating_condition']
    current_records = [
        record for record in state['records']
        if record.get('round_index') == round_index
    ]
    if any('stratum_key' not in record for record in current_records):
        raise ValueError('Existing round records are missing balanced stratum metadata')
    progress = {}
    for stratum_index, stratum_key in enumerate(sorted(strata)):
        records = sorted([
            record for record in current_records
            if tuple(record['stratum_key']) == stratum_key
        ], key=lambda record: (
            record['anchor_index'], record['noise_index'], record['request_id']
        ))
        accepted_count = 0
        for record in records:
            if record['status'] != 'success':
                continue
            if accepted_count < target_success_count:
                record['quota_status'] = 'accepted'
                accepted_count += 1
            else:
                record['quota_status'] = 'excess'
                record['status'] = 'quota_excess_success'
        accepted = [record for record in records if record['status'] == 'success']
        anchor_indices = [record['anchor_index'] for record in records]
        progress[stratum_key] = {
            'stratum_index': stratum_index,
            'success_count': len(accepted),
            'attempt_count': len(records),
            'next_anchor_index': max(anchor_indices) + 1 if anchor_indices else 0,
        }
    save_accumulated_dataset(ACCUMULATED_DATASET_PATH, state)
    collector = RoundRecordCollector(
        config,
        alternating_config,
        state,
        generator_checkpoint_path,
        round_index,
        device,
    )
    checkpoint_interval = alternating_config['checkpoint_interval_collections']
    completed_collections = 0
    last_saved_collection = 0
    while True:
        exhausted = [
            key for key, values in progress.items()
            if values['success_count'] < target_success_count
            and values['attempt_count'] >= max_attempt_count
        ]
        if exhausted:
            details = [
                {
                    'alpha': key[0],
                    'Re': key[1],
                    'success_count': progress[key]['success_count'],
                    'attempt_count': progress[key]['attempt_count'],
                }
                for key in exhausted
            ]
            raise RuntimeError(
                'XFoil success quota was not met before the per-condition attempt '
                f'limit: {details}'
            )
        active_keys = [
            key for key in sorted(strata)
            if progress[key]['success_count'] < target_success_count
        ]
        if not active_keys:
            break
        candidate_requests = []
        request_keys = []
        for stratum_key in active_keys:
            values = progress[stratum_key]
            request_result = build_stratum_candidate_requests(
                config,
                alternating_config,
                round_index,
                values['stratum_index'],
                stratum_key,
                strata[stratum_key],
                values['next_anchor_index'],
                max_attempt_count - values['attempt_count'],
            )
            requests, _sampled = request_result
            candidate_requests.extend(requests)
            request_keys.extend([stratum_key] * len(requests))
            values['attempt_count'] += len(requests)
            values['next_anchor_index'] += 1
        records = collector.collect(candidate_requests)
        for record, stratum_key in zip(records, request_keys):
            if record['status'] != 'success':
                continue
            values = progress[stratum_key]
            if values['success_count'] < target_success_count:
                record['quota_status'] = 'accepted'
                values['success_count'] += 1
            else:
                record['quota_status'] = 'excess'
                record['status'] = 'quota_excess_success'
        completed_collections += 1
        if completed_collections % checkpoint_interval == 0:
            save_accumulated_dataset(ACCUMULATED_DATASET_PATH, state)
            last_saved_collection = completed_collections
        completed = sum(values['success_count'] for values in progress.values())
        total = target_success_count * len(progress)
        print(f'Round {round_index + 1}: accepted {completed}/{total} balanced XFoil samples')
    if completed_collections != last_saved_collection:
        save_accumulated_dataset(ACCUMULATED_DATASET_PATH, state)
    round_records = [
        record for record in state['records']
        if record.get('round_index') == round_index
    ]
    status_counts = {}
    for record in round_records:
        status = record['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    per_stratum = [
        {
            'alpha': key[0],
            'Re': key[1],
            'successful_count': values['success_count'],
            'attempt_count': values['attempt_count'],
        }
        for key, values in sorted(progress.items())
    ]
    attempted_count = sum(values['attempt_count'] for values in progress.values())
    sampled_by_anchor = {}
    for record in round_records:
        anchor_key = (tuple(record['stratum_key']), record['anchor_index'])
        sampled_by_anchor.setdefault(anchor_key, {
            'condition': record['requested_condition'].clone(),
            'source': record['condition_source'],
            'mahalanobis_radius': record['mahalanobis_radius'],
        })
    xfoil_success_count = (
        status_counts.get('success', 0)
        + status_counts.get('quota_excess_success', 0)
    )
    return round_records, list(sampled_by_anchor.values()), build_condition_statistics(strata), {
        'successful_samples_per_operating_condition': target_success_count,
        'max_xfoil_attempts_per_operating_condition': max_attempt_count,
        'operating_condition_count': len(progress),
        'requested_count': attempted_count,
        'successful_count': target_success_count * len(progress),
        'xfoil_attempted_count': attempted_count,
        'convergence_rate': xfoil_success_count / attempted_count if attempted_count else 0.0,
        'status_counts': status_counts,
        'per_operating_condition': per_stratum,
    }


def _sample_dataset_batch(dataset, batch_size):
    indices = torch.randint(len(dataset), (batch_size,), device=dataset.coords.device)
    return dataset.coords[indices], dataset.conditions[indices], dataset.targets[indices]


def _allocate_counts(batch_size, ratio, first_available, second_available):
    if not first_available and not second_available:
        raise ValueError('Cannot sample a batch from two empty datasets')
    if not first_available:
        return 0, batch_size
    if not second_available:
        return batch_size, 0
    first_count = int(round(batch_size * ratio[0] / sum(ratio)))
    first_count = min(max(first_count, 1), batch_size - 1)
    return first_count, batch_size - first_count


def _sample_mixed_generated_batch(history_dataset, new_dataset, batch_size, ratio):
    history_count, new_count = _allocate_counts(
        batch_size,
        ratio,
        history_dataset is not None,
        new_dataset is not None,
    )
    batches = []
    if history_count:
        batches.append(_sample_dataset_batch(history_dataset, history_count))
    if new_count:
        batches.append(_sample_dataset_batch(new_dataset, new_count))
    return tuple(torch.cat([batch[index] for batch in batches], dim=0) for index in range(3))


def _save_surrogate_checkpoint(model, path, round_index, baseline_path, config):
    _require_parent_directory(path, 'Alternating surrogate checkpoint path')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'selection_policy': ALTERNATING_CHECKPOINT_POLICY,
        'training_epoch_count': config[ALTERNATING_CONFIG_KEY]['surrogate_epochs_per_round'],
        'round_index': round_index,
        'baseline_checkpoint_path': baseline_path,
        'target_names': SURROGATE_TARGET_ORDER,
        'condition_names': ['alpha', 'Re'],
        'target_loss_weights': config['surrogate_target_loss_weights'],
    }, path)


def _load_surrogate(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint['target_names'] != SURROGATE_TARGET_ORDER:
        raise ValueError(f'Unexpected surrogate target order in {checkpoint_path}')
    if checkpoint['condition_names'] != ['alpha', 'Re']:
        raise ValueError(f'Unexpected surrogate condition order in {checkpoint_path}')
    model = AerodynamicSurrogate(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def finetune_surrogate_round(
    config,
    alternating_config,
    raw_data,
    manifest,
    history_records,
    new_records,
    baseline_checkpoint_path,
    round_index,
    device,
):
    new_success = successful_records(new_records)
    if not new_success:
        raise RuntimeError('Cannot fine-tune surrogate because this round has no XFoil successes')
    auxiliary_stats = load_gan_auxiliary_stats(config, device)
    history_success = successful_records(history_records)
    history_dataset = (
        GeneratedSurrogateDataset(history_success, auxiliary_stats, device)
        if history_success else None
    )
    new_dataset = GeneratedSurrogateDataset(new_success, auxiliary_stats, device)
    original_dataset = AirfoilSurrogateDataset.from_norm_path(
        raw_data, SURROGATE_NORM_PATH, device
    )
    development_indices = original_dataset.prepare_indices(manifest['development_indices'])
    test_indices = original_dataset.prepare_indices(manifest['test_indices'])
    criterion = build_weighted_mse_loss(config, device)
    model = _load_surrogate(config, baseline_checkpoint_path, device)
    before_test = evaluate_generated_surrogate_metrics(
        model,
        original_dataset,
        test_indices,
        criterion,
        alternating_config['surrogate_original_replay_batch_size'],
        device,
    )
    baseline_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(alternating_config['surrogate_learning_rate']),
        weight_decay=float(config['surrogate_weight_decay']),
    )
    generated_count = len(history_success) + len(new_success)
    steps_per_epoch = max(
        math.ceil(generated_count / alternating_config['surrogate_generated_batch_size']),
        math.ceil(len(development_indices) / alternating_config['surrogate_original_replay_batch_size']),
    )
    final_epoch_metrics = None
    for epoch in range(alternating_config['surrogate_epochs_per_round']):
        model.train()
        loss_sums = {'generated': 0.0, 'original': 0.0, 'anchor': 0.0, 'total': 0.0}
        for _ in range(steps_per_epoch):
            generated_batch = _sample_mixed_generated_batch(
                history_dataset,
                new_dataset,
                alternating_config['surrogate_generated_batch_size'],
                alternating_config['surrogate_historical_to_new_ratio'],
            )
            positions = torch.randint(
                development_indices.numel(),
                (alternating_config['surrogate_original_replay_batch_size'],),
                device=device,
            )
            original_batch_indices = development_indices[positions]
            original_batch = (
                original_dataset.coords[original_batch_indices],
                original_dataset.conditions[original_batch_indices],
                original_dataset.targets[original_batch_indices],
            )
            optimizer.zero_grad(set_to_none=True)
            generated_loss = criterion(
                model(generated_batch[0], generated_batch[1]), generated_batch[2]
            )
            original_loss = criterion(
                model(original_batch[0], original_batch[1]), original_batch[2]
            )
            anchor_loss = torch.zeros((), device=device)
            if float(alternating_config['surrogate_lambda_anchor']) > 0.0:
                for parameter, baseline in zip(model.parameters(), baseline_parameters):
                    anchor_loss = anchor_loss + torch.sum((parameter - baseline) ** 2)
            total_loss = (
                float(alternating_config['surrogate_lambda_generated']) * generated_loss
                + float(alternating_config['surrogate_lambda_original']) * original_loss
                + float(alternating_config['surrogate_lambda_anchor']) * anchor_loss
            )
            total_loss.backward()
            optimizer.step()
            loss_sums['generated'] += float(generated_loss.detach().item())
            loss_sums['original'] += float(original_loss.detach().item())
            loss_sums['anchor'] += float(anchor_loss.detach().item())
            loss_sums['total'] += float(total_loss.detach().item())
        metrics = {key: value / steps_per_epoch for key, value in loss_sums.items()}
        final_epoch_metrics = metrics
        print(
            f'Round {round_index + 1} surrogate epoch '
            f'{epoch + 1}/{alternating_config["surrogate_epochs_per_round"]}: '
            f'generated={metrics["generated"]:.6f} '
            f'original={metrics["original"]:.6f} '
            f'anchor={metrics["anchor"]:.6f} total={metrics["total"]:.6f}'
        )
    output_path = _round_path(
        ALTERNATING_CHECKPOINT_DIRECTORY, round_index, 'surrogate.pt'
    )
    _save_surrogate_checkpoint(model, output_path, round_index, baseline_checkpoint_path, config)
    after_test = evaluate_generated_surrogate_metrics(
        model,
        original_dataset,
        test_indices,
        criterion,
        alternating_config['surrogate_original_replay_batch_size'],
        device,
    )
    all_records = history_success + new_success
    all_dataset = GeneratedSurrogateDataset(all_records, auxiliary_stats, device)
    all_indices = all_dataset.prepare_indices(list(range(len(all_dataset))))
    generated_training_metrics = evaluate_generated_surrogate_metrics(
        model,
        all_dataset,
        all_indices,
        criterion,
        alternating_config['surrogate_generated_batch_size'],
        device,
    )
    return output_path, {
        'checkpoint_path': output_path,
        'baseline_checkpoint_path': baseline_checkpoint_path,
        'epochs': alternating_config['surrogate_epochs_per_round'],
        'historical_success_count': len(history_success),
        'new_success_count': len(new_success),
        'generated_training_count': len(all_records),
        'final_epoch_losses': final_epoch_metrics,
        'original_independent_test': {
            'before': before_test,
            'after': after_test,
        },
        'generated_training_in_sample': generated_training_metrics,
    }


class GANRealPool:
    def __init__(self, coords, conditions, auxiliary_stats, device):
        if not coords:
            raise ValueError('GAN real pool coordinates must not be empty')
        physical_coords = torch.stack(coords).float().to(device)
        physical_conditions = torch.stack(conditions).float().to(device)
        self.coords = normalize_gan_coords(physical_coords, auxiliary_stats['gan_coord'])
        self.conditions = (
            physical_conditions - auxiliary_stats['gan_cond_mean']
        ) / auxiliary_stats['gan_cond_std']

    def __len__(self):
        return self.coords.size(0)

    def sample(self, count):
        indices = torch.randint(len(self), (count,), device=self.coords.device)
        return self.coords[indices], self.conditions[indices]


def build_gan_real_pools(raw_data, development_indices, generated_records, auxiliary_stats, device):
    original_pool = GANRealPool(
        [raw_data[index]['x'].float().view(-1, 2) for index in development_indices],
        [raw_data[index]['y'].float() for index in development_indices],
        auxiliary_stats,
        device,
    )
    successes = successful_records(generated_records)
    generated_pool = None
    if successes:
        generated_pool = GANRealPool(
            [record['coords'].float() for record in successes],
            [_actual_condition(record) for record in successes],
            auxiliary_stats,
            device,
        )
    return original_pool, generated_pool


def _sample_mixed_real_batch(original_pool, generated_pool, batch_size, ratio):
    original_count, generated_count = _allocate_counts(
        batch_size, ratio, original_pool is not None, generated_pool is not None
    )
    pieces = []
    if original_count:
        pieces.append(original_pool.sample(original_count))
    if generated_count:
        pieces.append(generated_pool.sample(generated_count))
    return (
        torch.cat([piece[0] for piece in pieces], dim=0),
        torch.cat([piece[1] for piece in pieces], dim=0),
    )


def train_gan_round(
    config,
    alternating_config,
    raw_data,
    manifest,
    generated_records,
    extended_conditions,
    gan_checkpoint_path,
    surrogate_checkpoint_path,
    round_index,
    device,
):
    checkpoint = torch.load(gan_checkpoint_path, map_location=device, weights_only=True)
    required = ('generator_state_dict', 'discriminator_state_dict')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'GAN checkpoint is missing keys: {missing}')
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    surrogate = _load_surrogate(config, surrogate_checkpoint_path, device)
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    auxiliary_stats = load_gan_auxiliary_stats(config, device)
    original_pool, generated_pool = build_gan_real_pools(
        raw_data,
        manifest['development_indices'],
        generated_records,
        auxiliary_stats,
        device,
    )
    physical_extended_conditions = torch.stack([
        sample['condition'] for sample in extended_conditions
    ]).to(device)
    normalized_extended_conditions = (
        physical_extended_conditions - auxiliary_stats['gan_cond_mean']
    ) / auxiliary_stats['gan_cond_std']
    optimizer_generator = torch.optim.Adam(
        generator.parameters(),
        lr=float(alternating_config['gan_learning_rate']),
        betas=(0.0, 0.9),
        weight_decay=5e-5,
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(alternating_config['gan_learning_rate']),
        betas=(0.0, 0.9),
        weight_decay=5e-5,
    )
    real_count = len(original_pool) + (len(generated_pool) if generated_pool else 0)
    steps_per_epoch = math.ceil(real_count / alternating_config['gan_batch_size'])
    n_critic = config['n_critic']
    final_epoch_metrics = None
    for epoch in range(alternating_config['gan_epochs_per_round']):
        sums = {
            'd_loss': 0.0,
            'g_adv': 0.0,
            'g_aero': 0.0,
            'g_trailing_edge_crossing': 0.0,
            'g_total': 0.0,
        }
        generator_steps = 0
        for step in range(steps_per_epoch):
            real_foils, real_conditions = _sample_mixed_real_batch(
                original_pool,
                generated_pool,
                alternating_config['gan_batch_size'],
                alternating_config['gan_real_original_to_generated_ratio'],
            )
            optimizer_discriminator.zero_grad(set_to_none=True)
            noise = torch.randn(real_foils.size(0), config['noise_dimension'], device=device)
            fake_foils = generator(noise, real_conditions)
            real_score = discriminator(real_foils, real_conditions)
            fake_score = discriminator(fake_foils.detach(), real_conditions)
            gradient_penalty, _ = compute_gradient_penalty(
                discriminator, real_foils, fake_foils.detach(), real_conditions, device
            )
            d_loss = -real_score.mean() + fake_score.mean() + config['lambda_gp'] * gradient_penalty
            d_loss.backward()
            optimizer_discriminator.step()
            sums['d_loss'] += float(d_loss.detach().item())
            if step % n_critic != 0:
                continue
            optimizer_generator.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            adv_noise = torch.randn(real_foils.size(0), config['noise_dimension'], device=device)
            adv_fake_foils, adv_trailing_edge_crossing_loss = (
                generator.generate_with_trailing_edge_crossing_loss(
                    adv_noise,
                    real_conditions,
                )
            )
            adversarial_loss = -discriminator(adv_fake_foils, real_conditions).mean()
            ext_indices = torch.randint(
                normalized_extended_conditions.size(0),
                (real_foils.size(0),),
                device=device,
            )
            ext_conditions = normalized_extended_conditions[ext_indices]
            ext_noise = torch.randn(real_foils.size(0), config['noise_dimension'], device=device)
            ext_fake_foils, ext_trailing_edge_crossing_loss = (
                generator.generate_with_trailing_edge_crossing_loss(
                    ext_noise,
                    ext_conditions,
                )
            )
            aerodynamic_loss, _, _ = compute_generator_auxiliary_losses(
                ext_fake_foils, ext_conditions, surrogate, auxiliary_stats, config
            )
            trailing_edge_crossing_loss = 0.5 * (
                adv_trailing_edge_crossing_loss + ext_trailing_edge_crossing_loss
            )
            total_loss = (
                float(alternating_config['gan_adversarial_weight']) * adversarial_loss
                + float(alternating_config['gan_aerodynamic_weight']) * aerodynamic_loss
                + trailing_edge_crossing_loss
            )
            total_loss.backward()
            optimizer_generator.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            sums['g_adv'] += float(adversarial_loss.detach().item())
            sums['g_aero'] += float(aerodynamic_loss.detach().item())
            sums['g_trailing_edge_crossing'] += float(
                trailing_edge_crossing_loss.detach().item()
            )
            sums['g_total'] += float(total_loss.detach().item())
            generator_steps += 1
        metrics = {
            'discriminator_loss': sums['d_loss'] / steps_per_epoch,
            'generator_adversarial_loss': sums['g_adv'] / generator_steps,
            'generator_aerodynamic_loss': sums['g_aero'] / generator_steps,
            'generator_trailing_edge_crossing_loss': (
                sums['g_trailing_edge_crossing'] / generator_steps
            ),
            'generator_total_loss': sums['g_total'] / generator_steps,
        }
        final_epoch_metrics = metrics
        print(
            f'Round {round_index + 1} GAN epoch '
            f'{epoch + 1}/{alternating_config["gan_epochs_per_round"]}: '
            f'D={metrics["discriminator_loss"]:.6f} '
            f'G_adv={metrics["generator_adversarial_loss"]:.6f} '
            f'G_aero={metrics["generator_aerodynamic_loss"]:.6f} '
            f'G_TE_cross={metrics["generator_trailing_edge_crossing_loss"]:.6f} '
            f'G_total={metrics["generator_total_loss"]:.6f}'
        )
    output_path = _round_path(
        ALTERNATING_CHECKPOINT_DIRECTORY, round_index, 'gan.pt'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_checkpoint(generator, discriminator, output_path)
    return output_path, {
        'checkpoint_path': output_path,
        'baseline_checkpoint_path': gan_checkpoint_path,
        'frozen_surrogate_checkpoint_path': surrogate_checkpoint_path,
        'epochs': alternating_config['gan_epochs_per_round'],
        'original_real_count': len(original_pool),
        'generated_real_count': len(generated_pool) if generated_pool else 0,
        'extended_condition_count': len(extended_conditions),
        'final_epoch_losses': final_epoch_metrics,
    }


def _coefficient_values_from_original(raw_data, development_indices):
    conditions = torch.stack([raw_data[index]['y'].float() for index in development_indices])
    cd = torch.tensor([raw_data[index]['cd'] for index in development_indices], dtype=torch.float32)
    return (
        conditions[:, 0].numpy(),
        conditions[:, 1].numpy(),
        conditions[:, 2].numpy(),
        conditions[:, 3].numpy(),
        cd.numpy(),
    )


def _coefficient_values_from_generated(records):
    successes = successful_records(records)
    if not successes:
        return tuple(np.empty(0, dtype=np.float64) for _ in range(5))
    conditions = torch.stack([_actual_condition(record) for record in successes])
    targets = torch.stack([record['targets'].float() for record in successes])
    return (
        conditions[:, 0].numpy(),
        conditions[:, 1].numpy(),
        conditions[:, 2].numpy(),
        conditions[:, 3].numpy(),
        targets[:, 2].numpy(),
    )


def plot_round_coefficients(config, raw_data, development_indices, history_records, new_records, path):
    existing_values = _coefficient_values_from_original(raw_data, development_indices)
    history_values = _coefficient_values_from_generated(history_records)
    new_values = _coefficient_values_from_generated(new_records)
    all_cd = np.concatenate([existing_values[4], history_values[4], new_values[4]])
    if all_cd.size == 0:
        raise ValueError('Cannot plot alternating coefficients without data')
    all_alpha = np.concatenate([existing_values[0], history_values[0], new_values[0]])
    all_reynolds = np.concatenate([existing_values[1], history_values[1], new_values[1]])
    alpha_values = np.unique(all_alpha)
    reynolds_values = np.unique(all_reynolds)
    color_scale = Normalize(vmin=float(all_cd.min()), vmax=float(all_cd.max()))
    figure, axes = plt.subplots(
        len(reynolds_values),
        len(alpha_values),
        figsize=(max(15, 3.2 * len(alpha_values)), max(12, 2.7 * len(reynolds_values))),
        squeeze=False,
    )
    scatter = None
    for row, reynolds_value in enumerate(reynolds_values):
        for column, alpha_value in enumerate(alpha_values):
            axis = axes[row, column]
            plotted = []
            point_counts = {
                'Existing original': 0,
                'Existing generated': 0,
                'New XFoil success': 0,
            }
            for values, label, marker, size, alpha in (
                (existing_values, 'Existing original', 'o', 10, 0.22),
                (history_values, 'Existing generated', 'o', 13, 0.45),
                (new_values, 'New XFoil success', 'X', 38, 0.9),
            ):
                mask = np.isclose(values[0], alpha_value) & np.isclose(values[1], reynolds_value)
                point_counts[label] = int(mask.sum())
                if not np.any(mask):
                    continue
                scatter_kwargs = {
                    'c': values[4][mask],
                    'cmap': 'viridis',
                    'norm': color_scale,
                    'marker': marker,
                    's': size,
                    'alpha': alpha,
                    'linewidths': 0,
                    'label': label,
                }
                if label == 'New XFoil success':
                    scatter_kwargs['edgecolors'] = 'black'
                    scatter_kwargs['linewidths'] = 0.65
                scatter = axis.scatter(
                    values[3][mask], values[2][mask],
                    **scatter_kwargs,
                )
                plotted.append((values[3][mask], values[2][mask]))
            reynolds_E5 = reynolds_value / 1e6
            axis.set_title(
                f'$\\alpha$ = {alpha_value:g} deg, Re = {reynolds_E5:g}E5'
            )
            existing_count = (
                point_counts['Existing original']
                + point_counts['Existing generated']
            )
            new_count = point_counts['New XFoil success']
            axis.text(
                0.02,
                0.03,
                f'Existing: {existing_count}\nNew: {new_count}',
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=8,
                bbox={
                    'boxstyle': 'round,pad=0.2',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.7,
                },
            )
            axis.grid(True, linestyle='--', alpha=0.35)
            if plotted:
                cm = np.concatenate([item[0] for item in plotted])
                cl = np.concatenate([item[1] for item in plotted])
                cm_padding = max((cm.max() - cm.min()) * 0.05, 1e-4)
                cl_padding = max((cl.max() - cl.min()) * 0.05, 1e-4)
                axis.set_xlim(float(cm.min() - cm_padding), float(cm.max() + cm_padding))
                axis.set_ylim(float(cl.min() - cl_padding), float(cl.max() + cl_padding))
            if row == len(reynolds_values) - 1:
                axis.set_xlabel('Cm')
            if column == 0:
                axis.set_ylabel('Cl')
    figure.suptitle('Alternating Fine-Tuning: Existing vs New XFoil Samples', y=0.995)
    figure.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.92, wspace=0.15, hspace=0.2)
    if scatter is not None:
        figure.colorbar(scatter, ax=axes.ravel().tolist(), label='Cd', pad=0.03)
    legend_entries = {}
    for axis in axes.ravel():
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_entries:
                legend_entries[label] = handle
    handles = list(legend_entries.values())
    labels = list(legend_entries)
    if handles:
        figure.legend(handles, labels, loc='upper center', ncol=3)
    save_report_figure(figure, path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def run_alternating_finetuning(config_path, round_count=None):
    config = load_config(config_path)
    alternating_config = resolve_alternating_config(config)
    configured_round_count = len(alternating_config['mahalanobis_radii'])
    if round_count is None:
        round_count = configured_round_count
    if not isinstance(round_count, int) or round_count <= 0 or round_count > configured_round_count:
        raise ValueError(f'round_count must be in [1, {configured_round_count}]')
    if not isinstance(config['max_workers'], int) or config['max_workers'] <= 0:
        raise ValueError('max_workers must be a positive integer')
    device = resolve_device(config)
    set_training_seed(config['surrogate_seed'])
    raw_data = torch.load(SURROGATE_DATASET_PATH, map_location='cpu', weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    state = load_accumulated_dataset(ACCUMULATED_DATASET_PATH)
    reports = []
    for round_index in range(round_count):
        if round_index in state['completed_round_indices']:
            print(f'Round {round_index + 1} is already complete; skipping')
            continue
        previous_gan_path = (
            INITIAL_GAN_CHECKPOINT_PATH
            if round_index == 0
            else _round_path(
                ALTERNATING_CHECKPOINT_DIRECTORY, round_index - 1, 'gan.pt'
            )
        )
        previous_surrogate_path = (
            INITIAL_SURROGATE_CHECKPOINT_PATH
            if round_index == 0
            else _round_path(
                ALTERNATING_CHECKPOINT_DIRECTORY, round_index - 1, 'surrogate.pt'
            )
        )
        if not os.path.exists(previous_gan_path):
            raise FileNotFoundError(f'GAN checkpoint does not exist: {previous_gan_path}')
        if not os.path.exists(previous_surrogate_path):
            raise FileNotFoundError(f'Surrogate checkpoint does not exist: {previous_surrogate_path}')
        history_before_round = [
            record for record in state['records']
            if record.get('round_index', -1) < round_index
        ]
        known_condition_records = build_known_condition_records(
            raw_data,
            manifest['development_indices'],
            history_before_round,
        )
        (
            round_records,
            extended_conditions,
            condition_statistics,
            collection_report,
        ) = collect_balanced_round_records(
            config,
            alternating_config,
            state,
            previous_gan_path,
            round_index,
            known_condition_records,
            device,
        )
        all_success_after_collection = successful_records(state['records'])
        surrogate_path, surrogate_report = finetune_surrogate_round(
            config,
            alternating_config,
            raw_data,
            manifest,
            history_before_round,
            round_records,
            previous_surrogate_path,
            round_index,
            device,
        )
        gan_path, gan_report = train_gan_round(
            config,
            alternating_config,
            raw_data,
            manifest,
            all_success_after_collection,
            extended_conditions,
            previous_gan_path,
            surrogate_path,
            round_index,
            device,
        )
        plot_path = _round_path(ALTERNATING_PLOT_DIRECTORY, round_index, 'coefficients.png')
        plot_round_coefficients(
            config,
            raw_data,
            manifest['development_indices'],
            history_before_round,
            round_records,
            plot_path,
        )
        report = {
            'round_index': round_index,
            'existing_to_sampled_ratio': alternating_config['existing_to_sampled_ratios'][round_index],
            'mahalanobis_radius': float(alternating_config['mahalanobis_radii'][round_index]),
            'condition_statistics': condition_statistics,
            'collection': collection_report,
            'surrogate': surrogate_report,
            'gan': gan_report,
            'coefficient_plot_path': plot_path,
            'accumulated_success_count': len(all_success_after_collection),
        }
        report_path = _round_path(ALTERNATING_REPORT_DIRECTORY, round_index, 'report.yaml')
        save_yaml(report_path, report)
        state['completed_round_indices'].append(round_index)
        save_accumulated_dataset(ACCUMULATED_DATASET_PATH, state)
        reports.append(report)
        print(f'Round {round_index + 1} complete: report saved to {report_path}')
    return reports


def main():
    parser = argparse.ArgumentParser(
        description='Alternately fine-tune the airfoil surrogate and CWGAN-GP'
    )
    parser.add_argument('--config', default='config.yaml', help='YAML configuration path')
    parser.add_argument(
        '--round-count', type=int, default=None,
        help='Number of configured rounds to execute, starting from round one',
    )
    arguments = parser.parse_args()
    run_alternating_finetuning(arguments.config, arguments.round_count)


if __name__ == '__main__':
    main()
