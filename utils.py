import numpy as np
import torch


def normalize_airfoil_chord_coordinates(coords, leading_edge_indices=None):
    """Transform Selig coordinates into the local unit-chord frame."""
    return_numpy = isinstance(coords, np.ndarray)
    if return_numpy:
        coordinates = torch.from_numpy(np.asarray(coords, dtype=float))
    elif isinstance(coords, torch.Tensor):
        if not coords.is_floating_point():
            raise TypeError('PyTorch coords must use a floating-point dtype')
        coordinates = coords
    else:
        raise TypeError(
            f'coords must be a NumPy array or PyTorch tensor, got {type(coords).__name__}'
        )

    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)
        single_airfoil = True
    elif coordinates.ndim == 3:
        single_airfoil = False
    else:
        raise ValueError(
            f'coords must have shape (N, 2) or (B, N, 2), got {tuple(coordinates.shape)}'
        )
    if coordinates.shape[-1] != 2:
        raise ValueError(
            f'coords must have shape (N, 2) or (B, N, 2), got {tuple(coordinates.shape)}'
        )
    if coordinates.shape[1] < 3:
        raise ValueError(
            f'coords must contain at least 3 points, got {coordinates.shape[1]}'
        )
    if not bool(torch.isfinite(coordinates).all().item()):
        raise ValueError('coords must contain only finite values')

    batch_size, point_count, _ = coordinates.shape
    if leading_edge_indices is None:
        leading_indices = torch.argmin(coordinates[:, :, 0], dim=1)
    else:
        leading_indices = torch.as_tensor(
            leading_edge_indices,
            dtype=torch.long,
            device=coordinates.device,
        )
        if leading_indices.ndim == 0:
            leading_indices = leading_indices.expand(batch_size)
        elif leading_indices.ndim != 1 or leading_indices.numel() != batch_size:
            raise ValueError(
                'leading_edge_indices must be a scalar or contain one index per airfoil'
            )
    if bool(torch.any(leading_indices <= 0).item()) or bool(
        torch.any(leading_indices >= point_count - 1).item()
    ):
        raise ValueError('leading edge must split non-empty upper and lower surfaces')

    batch_indices = torch.arange(batch_size, device=coordinates.device)
    leading_edges = coordinates[batch_indices, leading_indices]
    trailing_edge_midpoints = 0.5 * (coordinates[:, 0] + coordinates[:, -1])
    chord_vectors = trailing_edge_midpoints - leading_edges
    chord_lengths = torch.linalg.vector_norm(chord_vectors, dim=1)
    if bool(torch.any(chord_lengths <= 0.0).item()):
        raise ValueError('Airfoil chord length must be positive')

    chord_directions = chord_vectors / chord_lengths.unsqueeze(1)
    chord_normals = torch.stack(
        [-chord_directions[:, 1], chord_directions[:, 0]],
        dim=1,
    )
    relative_coordinates = coordinates - leading_edges.unsqueeze(1)
    normalized_x = torch.sum(
        relative_coordinates * chord_directions.unsqueeze(1),
        dim=2,
    ) / chord_lengths.unsqueeze(1)
    normalized_y = torch.sum(
        relative_coordinates * chord_normals.unsqueeze(1),
        dim=2,
    ) / chord_lengths.unsqueeze(1)

    leading_mask = torch.nn.functional.one_hot(
        leading_indices,
        num_classes=point_count,
    ).bool()
    trailing_mask = torch.zeros_like(leading_mask)
    trailing_mask[:, 0] = True
    trailing_mask[:, -1] = True
    normalized_x = torch.where(leading_mask, torch.zeros_like(normalized_x), normalized_x)
    normalized_x = torch.where(trailing_mask, torch.ones_like(normalized_x), normalized_x)
    normalized_y = torch.where(leading_mask, torch.zeros_like(normalized_y), normalized_y)
    normalized = torch.stack([normalized_x, normalized_y], dim=2)

    if single_airfoil:
        normalized = normalized[0]
    if return_numpy:
        return normalized.numpy()
    return normalized


def calculate_relative_thickness(coords):
    """Return max_x(y_upper(x) - y_lower(x)) / chord for Selig coordinates."""
    coordinates = np.asarray(coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f'coords must have shape (N, 2), got {coordinates.shape}'
        )
    if coordinates.shape[0] < 3:
        raise ValueError(
            f'coords must contain at least 3 points, got {coordinates.shape[0]}'
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('coords must contain only finite values')

    x = coordinates[:, 0]
    chord = float(np.max(x) - np.min(x))
    if chord <= 0:
        raise ValueError(f'chord must be positive, got {chord}')

    leading_edge_index = int(np.argmin(x))
    if leading_edge_index == 0 or leading_edge_index == coordinates.shape[0] - 1:
        raise ValueError('leading edge must split non-empty upper and lower surfaces')

    upper_x, upper_y = _prepare_surface(
        coordinates[:leading_edge_index + 1][::-1],
        'upper',
        np.maximum,
    )
    lower_x, lower_y = _prepare_surface(
        coordinates[leading_edge_index:],
        'lower',
        np.minimum,
    )
    shared_start = max(float(upper_x[0]), float(lower_x[0]))
    shared_end = min(float(upper_x[-1]), float(lower_x[-1]))
    if shared_start >= shared_end:
        raise ValueError('upper and lower surfaces have no shared chordwise interval')

    sample_x = np.unique(np.concatenate([
        upper_x[(upper_x >= shared_start) & (upper_x <= shared_end)],
        lower_x[(lower_x >= shared_start) & (lower_x <= shared_end)],
        np.array([shared_start, shared_end]),
    ]))
    thickness = np.interp(sample_x, upper_x, upper_y) - np.interp(
        sample_x,
        lower_x,
        lower_y,
    )
    maximum_thickness = float(np.max(thickness))
    if maximum_thickness < 0.0:
        raise ValueError('upper surface is below lower surface over the shared interval')

    return maximum_thickness / chord


def _prepare_surface(surface, surface_name, reducer):
    order = np.argsort(surface[:, 0], kind='stable')
    x_values = surface[order, 0]
    y_values = surface[order, 1]

    unique_x, inverse = np.unique(x_values, return_inverse=True)
    if unique_x.size < 2:
        raise ValueError(f'{surface_name} surface must span at least two x coordinates')
    if reducer is np.maximum:
        unique_y = np.full(unique_x.shape, -np.inf)
    else:
        unique_y = np.full(unique_x.shape, np.inf)
    reducer.at(unique_y, inverse, y_values)
    return unique_x, unique_y


def check_intersection(coords):
    """
    Check if the airfoil curve self-intersects.
    coords: (N, 2) array of coordinates.
    returns: True if self-intersects, False otherwise.
    """
    N = len(coords)
    if N < 4:
        return False
        
    A = coords[:-1]
    B = coords[1:]
    
    def ccw(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) - (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
    
    A_exp = A[:, None, :]
    B_exp = B[:, None, :]
    C_exp = A[None, :, :]
    D_exp = B[None, :, :]
    
    ccw1 = ccw(A_exp, C_exp, D_exp)
    ccw2 = ccw(B_exp, C_exp, D_exp)
    ccw3 = ccw(A_exp, B_exp, C_exp)
    ccw4 = ccw(A_exp, B_exp, D_exp)
    
    intersect = ((ccw1 * ccw2) < 0) & ((ccw3 * ccw4) < 0)
    
    # Check only non-adjacent segments: index j > i + 1
    mask = np.triu(np.ones((N-1, N-1), dtype=bool), k=2)
    
    return np.any(intersect & mask)

def check_shape_intersections(coords):
    """
    Check if the airfoil shape is valid based on ray intersections.
    - Any vertical line should intersect the curve at most 2 times.
    - Any horizontal line should intersect the curve at most 4 times.
    Returns True if the shape is INVALID (fails the check), False if valid.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # vertical lines
    x_sorted = np.unique(x)
    x_test = (x_sorted[:-1] + x_sorted[1:]) / 2.0
    x1 = x[:-1]
    x2 = x[1:]
    
    for c in x_test:
        intersections = np.sum((x1 - c) * (x2 - c) < 0)
        if intersections > 2:
            return True
            
    # horizontal lines
    y_sorted = np.unique(y)
    y_test = (y_sorted[:-1] + y_sorted[1:]) / 2.0
    y1 = y[:-1]
    y2 = y[1:]
    
    for c in y_test:
        intersections = np.sum((y1 - c) * (y2 - c) < 0)
        if intersections > 4:
            return True
            
    return False

