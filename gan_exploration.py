import math

import torch

from gan_conditions import GAN_LABEL_ORDER
from model import AerodynamicSurrogate, Generator
from surrogate_split import load_cross_validation_manifest
from train import (
    build_surrogate_conditions,
    denormalize_gan_coords,
    load_gan_auxiliary_stats,
    normalize_surrogate_coords,
)


EXPLORATION_CONFIG_KEY = 'gan_exploration'
EXPLORATION_CONFIG_KEYS = (
    'condition_count',
    'empirical_condition_fraction',
    'mahalanobis_radius',
    'noise_samples_per_condition',
    'generation_batch_size',
    'xfoil_timeout_seconds',
    'latent_optimization_start_count',
    'latent_optimization_steps',
    'latent_optimization_learning_rate',
    'latent_trust_radius',
    'cl_tolerance',
    'cm_tolerance',
    'cl_penalty_weight',
    'cm_penalty_weight',
)
SURROGATE_DATASET_PATH = 'model/airfoil_dataset.pt'
EXPLORATION_GAN_CHECKPOINT_PATH = 'model/gan_final.pt'
EXPLORATION_SURROGATE_CHECKPOINT_PATH = 'model/surrogate_airfoil_group_best.pt'
SURROGATE_TARGET_ORDER = ['CM', 'CL', 'CD']
SUPPORTED_SURROGATE_SELECTION_POLICIES = {
    'fixed_final_epoch',
    'generated_validation_weighted_mse',
    'alternating_fixed_epoch',
}


def resolve_exploration_config(config):
    if EXPLORATION_CONFIG_KEY not in config:
        raise ValueError(f'config is missing {EXPLORATION_CONFIG_KEY}')
    exploration_config = config[EXPLORATION_CONFIG_KEY]
    missing_keys = [
        key for key in EXPLORATION_CONFIG_KEYS if key not in exploration_config
    ]
    if missing_keys:
        raise ValueError(
            f'{EXPLORATION_CONFIG_KEY} is missing required keys: {missing_keys}'
        )
    positive_integer_keys = (
        'condition_count',
        'noise_samples_per_condition',
        'generation_batch_size',
        'xfoil_timeout_seconds',
        'latent_optimization_start_count',
        'latent_optimization_steps',
    )
    for key in positive_integer_keys:
        value = exploration_config[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f'{EXPLORATION_CONFIG_KEY}.{key} must be a positive integer'
            )
    fraction = float(exploration_config['empirical_condition_fraction'])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f'{EXPLORATION_CONFIG_KEY}.empirical_condition_fraction must be in [0, 1]'
        )
    positive_float_keys = (
        'mahalanobis_radius',
        'latent_optimization_learning_rate',
        'latent_trust_radius',
        'cl_tolerance',
        'cm_tolerance',
        'cl_penalty_weight',
        'cm_penalty_weight',
    )
    for key in positive_float_keys:
        if float(exploration_config[key]) <= 0.0:
            raise ValueError(f'{EXPLORATION_CONFIG_KEY}.{key} must be positive')
    return exploration_config


def load_development_condition_data(config):
    raw_data = torch.load(
        SURROGATE_DATASET_PATH, map_location='cpu', weights_only=True
    )
    manifest = load_cross_validation_manifest(raw_data, config)
    return raw_data, manifest['development_indices']


def _condition_stratum_key(condition):
    return float(condition[0].item()), float(condition[1].item())


def build_development_condition_strata(raw_data, development_indices):
    if not development_indices:
        raise ValueError('development_indices must not be empty')
    return build_condition_record_strata([
        {
            'record_id': dataset_index,
            'condition': raw_data[dataset_index]['y'].float(),
        }
        for dataset_index in development_indices
    ])


def build_condition_record_strata(condition_records):
    if not condition_records:
        raise ValueError('condition_records must not be empty')
    strata = {}
    for record in condition_records:
        if 'record_id' not in record or 'condition' not in record:
            raise ValueError('Condition record must contain record_id and condition')
        labels = record['condition'].float()
        if labels.ndim != 1 or labels.numel() != len(GAN_LABEL_ORDER):
            raise ValueError(
                f'Condition record {record["record_id"]!r} must contain '
                f'{GAN_LABEL_ORDER}, got shape {tuple(labels.shape)}'
            )
        if not bool(torch.isfinite(labels).all().item()):
            raise ValueError(f'Condition record {record["record_id"]!r} has non-finite labels')
        key = _condition_stratum_key(labels)
        strata.setdefault(key, []).append((record['record_id'], labels))
    for key, records in strata.items():
        if len(records) < 2:
            raise ValueError(
                f'Condition stratum {key} has fewer than two known samples'
            )
    return strata


def _pearson_correlation(values):
    if values.ndim != 2 or values.size(1) != 2:
        raise ValueError(f'Expected values with shape (N, 2), got {tuple(values.shape)}')
    if values.size(0) < 2:
        return None
    centered = values - values.mean(dim=0, keepdim=True)
    scale = torch.sqrt(torch.sum(centered.square(), dim=0))
    if bool(torch.any(scale == 0.0).item()):
        return None
    return float(torch.sum(centered[:, 0] * centered[:, 1]).item() / torch.prod(scale).item())


def _mean_covariance(values):
    if values.ndim != 2 or values.size(1) != 2 or values.size(0) < 2:
        raise ValueError('CL/CM values must have shape (N, 2) with N >= 2')
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T.matmul(centered) / (values.size(0) - 1)
    return mean, covariance


def build_condition_statistics(strata):
    if not strata:
        raise ValueError('Condition strata must not be empty')
    all_values = []
    within_values = []
    stratum_statistics = []
    for key in sorted(strata):
        records = strata[key]
        values = torch.stack([labels[[2, 3]] for _, labels in records]).float()
        mean, covariance = _mean_covariance(values)
        all_values.append(values)
        within_values.append(values - mean)
        stratum_statistics.append({
            'alpha': key[0],
            'Re': key[1],
            'sample_count': len(records),
            'CL_mean': float(mean[0].item()),
            'CM_mean': float(mean[1].item()),
            'CL_std': float(values[:, 0].std(unbiased=True).item()),
            'CM_std': float(values[:, 1].std(unbiased=True).item()),
            'CL_CM_covariance': float(covariance[0, 1].item()),
            'CL_CM_correlation': _pearson_correlation(values),
        })
    all_values = torch.cat(all_values, dim=0)
    within_values = torch.cat(within_values, dim=0)
    return {
        'label_order': GAN_LABEL_ORDER,
        'cl_cm_order': ['CL', 'CM'],
        'development_sample_count': int(all_values.size(0)),
        'stratum_count': len(strata),
        'global_CL_CM_correlation': _pearson_correlation(all_values),
        'within_stratum_CL_CM_correlation': _pearson_correlation(within_values),
        'strata': stratum_statistics,
    }


def _sample_covariance_condition(records, mahalanobis_radius, generator):
    values = torch.stack([labels[[2, 3]] for _, labels in records]).float()
    mean, covariance = _mean_covariance(values)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = torch.max(torch.abs(eigenvalues))
    tolerance = torch.finfo(values.dtype).eps * max(float(scale.item()), 1.0)
    positive_mask = eigenvalues > tolerance
    if not bool(torch.any(positive_mask).item()):
        raise ValueError('CL/CM covariance has zero numerical rank')
    normal = torch.randn(2, generator=generator)
    normal = normal * positive_mask.to(normal.dtype)
    normal_norm = torch.linalg.vector_norm(normal)
    if normal_norm == 0.0:
        normal[positive_mask.nonzero()[0, 0]] = 1.0
        normal_norm = torch.linalg.vector_norm(normal)
    direction = normal / normal_norm
    radial_scale = mahalanobis_radius * torch.sqrt(torch.rand((), generator=generator))
    whitened_delta = direction * radial_scale
    delta = eigenvectors.matmul(torch.sqrt(torch.clamp(eigenvalues, min=0.0)) * whitened_delta)
    return mean + delta, float(radial_scale.item())


def sample_condition_from_stratum(records, source, mahalanobis_radius, generator):
    if not records:
        raise ValueError('Condition stratum records must not be empty')
    if source not in ('empirical', 'covariance'):
        raise ValueError(f'Unsupported condition source: {source}')
    position = int(torch.randint(len(records), (), generator=generator).item())
    source_record_id, anchor_condition = records[position]
    condition = anchor_condition.float().clone()
    if source == 'empirical':
        return {
            'condition': condition,
            'source': source,
            'source_record_id': source_record_id,
            'mahalanobis_radius': 0.0,
        }
    cl_cm, radial_scale = _sample_covariance_condition(
        records, mahalanobis_radius, generator
    )
    condition[2:4] = cl_cm
    return {
        'condition': condition,
        'source': source,
        'source_record_id': source_record_id,
        'mahalanobis_radius': radial_scale,
    }


def sample_extended_conditions(
    raw_data,
    development_indices,
    condition_count,
    empirical_condition_fraction,
    mahalanobis_radius,
    seed,
):
    if not isinstance(condition_count, int) or condition_count <= 0:
        raise ValueError(f'condition_count must be a positive integer, got {condition_count}')
    if not 0.0 <= empirical_condition_fraction <= 1.0:
        raise ValueError('empirical_condition_fraction must be in [0, 1]')
    if mahalanobis_radius <= 0.0:
        raise ValueError('mahalanobis_radius must be positive')
    condition_records = [
        {
            'record_id': dataset_index,
            'condition': raw_data[dataset_index]['y'].float(),
        }
        for dataset_index in development_indices
    ]
    sampled, statistics = sample_extended_condition_records(
        condition_records,
        condition_count,
        empirical_condition_fraction,
        mahalanobis_radius,
        seed,
    )
    for sample in sampled:
        sample['source_dataset_index'] = sample.pop('source_record_id')
    return sampled, statistics


def sample_extended_condition_records(
    condition_records,
    condition_count,
    empirical_condition_fraction,
    mahalanobis_radius,
    seed,
):
    if not condition_records:
        raise ValueError('condition_records must not be empty')
    if not isinstance(condition_count, int) or condition_count <= 0:
        raise ValueError(f'condition_count must be a positive integer, got {condition_count}')
    if not 0.0 <= empirical_condition_fraction <= 1.0:
        raise ValueError('empirical_condition_fraction must be in [0, 1]')
    if mahalanobis_radius <= 0.0:
        raise ValueError('mahalanobis_radius must be positive')
    strata = build_condition_record_strata(condition_records)
    empirical_count = int(round(condition_count * empirical_condition_fraction))
    if empirical_count > len(condition_records):
        raise ValueError(
            f'Requested {empirical_count} empirical conditions, only '
            f'{len(condition_records)} known samples are available'
        )
    generator = torch.Generator().manual_seed(seed)
    sampled = []
    empirical_positions = torch.randperm(
        len(condition_records), generator=generator
    )[:empirical_count].tolist()
    for position in empirical_positions:
        source_record = condition_records[position]
        condition = source_record['condition'].float().clone()
        sampled.append({
            'condition': condition,
            'source': 'empirical',
            'source_record_id': source_record['record_id'],
            'mahalanobis_radius': 0.0,
        })
    for _ in range(condition_count - empirical_count):
        position = int(torch.randint(len(condition_records), (), generator=generator).item())
        source_record = condition_records[position]
        key = _condition_stratum_key(source_record['condition'])
        sampled.append(sample_condition_from_stratum(
            strata[key], 'covariance', mahalanobis_radius, generator
        ))
    if len(sampled) != condition_count:
        raise RuntimeError('Extended condition sampler returned an incorrect sample count')
    return sampled, build_condition_statistics(strata)


def load_exploration_models(config, device):
    exploration_config = resolve_exploration_config(config)
    generator_checkpoint = torch.load(
        EXPLORATION_GAN_CHECKPOINT_PATH, map_location=device, weights_only=True
    )
    if 'generator_state_dict' not in generator_checkpoint:
        raise ValueError('GAN exploration checkpoint is missing generator_state_dict')
    generator = Generator(config).to(device)
    generator.load_state_dict(generator_checkpoint['generator_state_dict'])
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    surrogate_checkpoint = torch.load(
        EXPLORATION_SURROGATE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    if surrogate_checkpoint['selection_policy'] not in SUPPORTED_SURROGATE_SELECTION_POLICIES:
        raise ValueError(
            f'Unsupported exploration surrogate selection policy: '
            f'{surrogate_checkpoint["selection_policy"]}'
        )
    if surrogate_checkpoint['condition_names'] != ['alpha', 'Re']:
        raise ValueError('Exploration surrogate must use [alpha, Re] conditions')
    if surrogate_checkpoint['target_names'] != SURROGATE_TARGET_ORDER:
        raise ValueError('Exploration surrogate must predict [CM, CL, CD]')
    surrogate = AerodynamicSurrogate(config).to(device)
    surrogate.load_state_dict(surrogate_checkpoint['model_state_dict'])
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    return generator, surrogate, load_gan_auxiliary_stats(config, device)


def generate_and_predict(generator, surrogate, auxiliary_stats, config, noise, conditions):
    if noise.ndim != 2 or conditions.ndim != 2:
        raise ValueError('noise and conditions must both have shape (batch, features)')
    if noise.size(0) != conditions.size(0):
        raise ValueError('noise and conditions must have the same batch size')
    if conditions.size(1) != len(GAN_LABEL_ORDER):
        raise ValueError(f'conditions must have {len(GAN_LABEL_ORDER)} columns')
    normalized_conditions = (
        conditions - auxiliary_stats['gan_cond_mean']
    ) / auxiliary_stats['gan_cond_std']
    generated_normalized_coords = generator(noise, normalized_conditions)
    physical_coords = denormalize_gan_coords(
        generated_normalized_coords,
        auxiliary_stats['gan_coord'],
        config['num_output_points'],
    )
    surrogate_coords = normalize_surrogate_coords(
        physical_coords, auxiliary_stats['surrogate_coord']
    )
    surrogate_conditions = build_surrogate_conditions(conditions, auxiliary_stats)
    normalized_predictions = surrogate(surrogate_coords, surrogate_conditions)
    predictions = (
        normalized_predictions * auxiliary_stats['surrogate_target_std']
        + auxiliary_stats['surrogate_target_mean']
    )
    return physical_coords, predictions


def optimize_latent_noise(
    generator,
    surrogate,
    auxiliary_stats,
    config,
    condition,
    seed,
):
    exploration_config = resolve_exploration_config(config)
    if condition.ndim != 1 or condition.numel() != len(GAN_LABEL_ORDER):
        raise ValueError(f'condition must contain {GAN_LABEL_ORDER}')
    device = next(generator.parameters()).device
    start_count = exploration_config['latent_optimization_start_count']
    random_generator = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn(
        start_count, config['noise_dimension'], generator=random_generator
    ).to(device)
    noise = torch.nn.Parameter(initial_noise.clone())
    conditions = condition.to(device).unsqueeze(0).expand(start_count, -1)
    optimizer = torch.optim.Adam(
        [noise], lr=float(exploration_config['latent_optimization_learning_rate'])
    )
    cl_index = SURROGATE_TARGET_ORDER.index('CL')
    cm_index = SURROGATE_TARGET_ORDER.index('CM')
    cd_index = SURROGATE_TARGET_ORDER.index('CD')
    for _ in range(exploration_config['latent_optimization_steps']):
        optimizer.zero_grad(set_to_none=True)
        _, predictions = generate_and_predict(
            generator, surrogate, auxiliary_stats, config, noise, conditions
        )
        cl_violation = torch.relu(
            torch.abs(predictions[:, cl_index] - conditions[:, 2])
            - float(exploration_config['cl_tolerance'])
        )
        cm_violation = torch.relu(
            torch.abs(predictions[:, cm_index] - conditions[:, 3])
            - float(exploration_config['cm_tolerance'])
        )
        loss = (
            predictions[:, cd_index].mean()
            + float(exploration_config['cl_penalty_weight']) * cl_violation.square().mean()
            + float(exploration_config['cm_penalty_weight']) * cm_violation.square().mean()
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta = noise - initial_noise
            delta_norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            projection = torch.clamp(
                float(exploration_config['latent_trust_radius']) / (delta_norm + 1e-12),
                max=1.0,
            )
            noise.copy_(initial_noise + delta * projection)
    with torch.no_grad():
        coords, predictions = generate_and_predict(
            generator, surrogate, auxiliary_stats, config, noise, conditions
        )
    return {
        'initial_noise': initial_noise.detach().cpu(),
        'optimized_noise': noise.detach().cpu(),
        'condition': condition.detach().cpu(),
        'coords': coords.detach().cpu(),
        'surrogate_predictions': predictions.detach().cpu(),
    }


def batch_random_noise(condition_count, noise_samples_per_condition, noise_dimension, seed):
    if condition_count <= 0 or noise_samples_per_condition <= 0 or noise_dimension <= 0:
        raise ValueError('condition_count, noise_samples_per_condition, and noise_dimension must be positive')
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        condition_count * noise_samples_per_condition,
        noise_dimension,
        generator=generator,
    )
