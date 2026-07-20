import math

import torch


def endpoint_dense_spacing(num_points, endpoint, beta):
    if num_points < 2:
        raise ValueError(f'num_points must be at least 2, got {num_points}')
    if beta <= 0.0:
        raise ValueError(f'beta must be positive, got {beta}')

    values = torch.linspace(0, 1, num_points)
    if endpoint == 'start':
        return values ** beta
    if endpoint == 'end':
        return 1.0 - (1.0 - values) ** beta
    raise ValueError(f"endpoint must be 'start' or 'end', got {endpoint}")


def split_surface_t_values(num_output_points, beta):
    if num_output_points < 3:
        raise ValueError(
            f'num_output_points must be at least 3, got {num_output_points}'
        )
    upper_output_points = num_output_points // 2 + 1
    lower_output_points = num_output_points - upper_output_points + 1
    return (
        endpoint_dense_spacing(upper_output_points, 'end', beta),
        endpoint_dense_spacing(lower_output_points, 'start', beta),
    )


def build_bernstein_basis(t_values, coefficient_count):
    if coefficient_count < 2:
        raise ValueError(
            f'coefficient_count must be at least 2, got {coefficient_count}'
        )
    degree = coefficient_count - 1
    t_double = t_values.to(torch.float64)
    basis = torch.zeros(
        (*t_values.shape, coefficient_count),
        dtype=torch.float64,
        device=t_values.device,
    )
    for index in range(coefficient_count):
        basis[..., index] = (
            math.comb(degree, index)
            * (t_double ** index)
            * ((1.0 - t_double) ** (degree - index))
        )
    return basis.to(torch.float32)


def bounded_cst_exponent(raw_value, value_range):
    lower, upper = value_range
    return lower + (upper - lower) * torch.sigmoid(raw_value)


def cst_surface_from_basis(basis, x_values, coefficients, trailing_edge_y, n1, n2):
    if basis.ndim == 2:
        shape = torch.matmul(
            basis.unsqueeze(0),
            coefficients.unsqueeze(-1),
        ).squeeze(-1)
    elif basis.ndim == 3:
        shape = torch.bmm(basis, coefficients.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f'basis must have 2 or 3 dimensions, got {basis.ndim}')
    class_function = x_values.pow(n1) * (1.0 - x_values).pow(n2)
    return class_function * shape + x_values * trailing_edge_y


def decode_split_surface_cst(
    upper_basis,
    lower_basis,
    upper_x,
    lower_x,
    upper_coefficients,
    lower_coefficients,
    upper_te_y,
    lower_te_y,
    n1,
    n2,
):
    upper_y = cst_surface_from_basis(
        upper_basis,
        upper_x,
        upper_coefficients,
        upper_te_y,
        n1,
        n2,
    )
    lower_y = cst_surface_from_basis(
        lower_basis,
        lower_x,
        lower_coefficients,
        lower_te_y,
        n1,
        n2,
    )
    upper_curve = torch.stack([upper_x.expand_as(upper_y), upper_y], dim=-1)
    lower_curve = torch.stack([lower_x.expand_as(lower_y), lower_y], dim=-1)
    return torch.cat([upper_curve, lower_curve[:, 1:, :]], dim=1)
