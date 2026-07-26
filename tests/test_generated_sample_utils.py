import torch

from generated_surrogate_utils import GeneratedSurrogateDataset
from generated_xfoil_utils import build_xfoil_cache_key, prepare_generated_xfoil_record


def make_coords():
    return torch.tensor([
        [1.0, 0.0],
        [0.5, 0.08],
        [0.0, 0.0],
        [0.5, -0.08],
        [1.0, 0.0],
    ])


def test_prepared_generated_xfoil_record_has_geometry_and_operating_cache_key():
    request = {
        'request_id': 'round_001:0:0',
        'source_dataset_index': 'original:3',
        'noise_index': 0,
        'condition': torch.tensor([4.0, 200000.0, 0.7, -0.05]),
    }
    record = prepare_generated_xfoil_record(request, make_coords(), 'round_001')

    assert record['status'] == 'pending'
    assert record['coords'].shape == (5, 2)
    assert record['cache_key'] == build_xfoil_cache_key(record['coords'], 4.0, 200000.0)


def test_generated_surrogate_dataset_normalizes_generated_records():
    records = [{
        'coords': make_coords(),
        'condition': torch.tensor([4.0, 200000.0, 0.7, -0.05]),
        'targets': torch.tensor([-0.06, 0.68, 0.012]),
    }]
    auxiliary_stats = {
        'surrogate_coord': {
            'y_min': torch.tensor(-0.1),
            'y_max': torch.tensor(0.1),
        },
        'surrogate_condition_mean': torch.tensor([4.0, 200000.0]),
        'surrogate_condition_std': torch.tensor([2.0, 100000.0]),
        'surrogate_target_mean': torch.tensor([0.0, 0.5, 0.01]),
        'surrogate_target_std': torch.tensor([0.1, 0.2, 0.01]),
    }

    dataset = GeneratedSurrogateDataset(records, auxiliary_stats, torch.device('cpu'))
    indices = dataset.prepare_indices([0])

    assert dataset.coords.shape == (1, 10)
    assert dataset.conditions.shape == (1, 2)
    assert torch.allclose(dataset.targets[0], torch.tensor([-0.6, 0.9, 0.2]))
    assert next(dataset.iter_batches(indices, batch_size=1, shuffle=False))[0].shape == (1, 10)
