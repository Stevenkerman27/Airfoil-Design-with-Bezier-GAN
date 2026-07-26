import hashlib
import struct

import numpy as np
import torch

from foildata.xfoil import run_xfoil_single
from utils import normalize_airfoil_chord_coordinates


GENERATED_XFOIL_TARGET_ORDER = ['CM', 'CL', 'CD']


def normalized_coordinate_hash(coords):
    array = np.ascontiguousarray(coords.detach().cpu().numpy().astype('<f4'))
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_xfoil_cache_key(coords, alpha, reynolds):
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(normalized_coordinate_hash(coords)))
    digest.update(struct.pack('<dd', float(alpha), float(reynolds)))
    return digest.hexdigest()


def prepare_generated_xfoil_record(request, generated_coords, collection_id):
    condition = request['condition']
    try:
        normalized_coords = normalize_airfoil_chord_coordinates(generated_coords).cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        return {
            'collection_id': collection_id,
            'request_id': request['request_id'],
            'source_dataset_index': request['source_dataset_index'],
            'noise_index': request['noise_index'],
            'condition': condition.cpu(),
            'status': 'preprocess_failed',
            'failure_reason': str(error),
        }
    alpha = float(condition[0].item())
    reynolds = float(condition[1].item())
    return {
        'collection_id': collection_id,
        'request_id': request['request_id'],
        'source_dataset_index': request['source_dataset_index'],
        'noise_index': request['noise_index'],
        'generation_id': hashlib.sha256(
            normalized_coords.numpy().astype('<f4').tobytes()
        ).hexdigest(),
        'cache_key': build_xfoil_cache_key(normalized_coords, alpha, reynolds),
        'coords': normalized_coords.float(),
        'condition': condition.cpu(),
        'status': 'pending',
    }


def run_xfoil_for_generated_record(record, timeout_seconds):
    if record['status'] != 'pending':
        return record
    alpha = float(record['condition'][0].item())
    reynolds = float(record['condition'][1].item())
    result = run_xfoil_single(
        record['coords'].numpy(),
        reynolds,
        alpha,
        timeout=timeout_seconds,
        return_all=True,
    )
    if result is None:
        record['status'] = 'xfoil_failed'
        return record
    try:
        targets = torch.tensor(
            [result['CM'], result['CL'], result['CD']], dtype=torch.float32
        )
    except (KeyError, TypeError, ValueError) as error:
        record['status'] = 'xfoil_failed'
        record['failure_reason'] = f'incomplete_xfoil_result: {error}'
        return record
    if not bool(torch.isfinite(targets).all().item()):
        record['status'] = 'xfoil_failed'
        record['failure_reason'] = 'non_finite_xfoil_result'
        return record
    record['status'] = 'success'
    record['targets'] = targets
    return record
