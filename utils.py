import numpy as np
import torch


def calculate_relative_thickness(coords):
    """
    Calculate relative thickness with the project-wide max-min definition.
    coords: (N, 2) array of coordinates.
    returns: rel_thickness (float)
    """
    x = coords[:, 0]
    y = coords[:, 1]

    chord = np.max(x) - np.min(x)
    if chord <= 0:
        return 0.0

    rel_thickness = (np.max(y) - np.min(y)) / chord
    return float(rel_thickness)


def calculate_relative_thickness_torch(coords):
    """
    Differentiable PyTorch version of calculate_relative_thickness.
    coords: (B, N, 2) tensor of physical coordinates.
    returns: (B,) tensor of relative thickness values.
    """
    if coords.dim() != 3 or coords.size(2) != 2:
        raise ValueError(f"coords must have shape (B, N, 2), got {tuple(coords.shape)}")

    x_values = coords[:, :, 0]
    y_values = coords[:, :, 1]
    chord = torch.amax(x_values, dim=1) - torch.amin(x_values, dim=1)
    height = torch.amax(y_values, dim=1) - torch.amin(y_values, dim=1)
    return height / (chord + 1e-8)


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

