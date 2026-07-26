# Airfoil Preprocessing

Processed airfoils use the same local chord frame as XFoil `NORM`: the minimum-x leading edge is `(0, 0)`, and the midpoint of the two trailing-edge points is `(1, 0)`.

## Fixed Chordwise Sampling

Each source airfoil is first transformed into the local unit-chord frame, then split at its validated leading-edge point. The upper and lower surfaces are sorted in increasing `x`. Repeated source `x` values are reduced to the maximum upper-surface `y` and minimum lower-surface `y`, respectively.

The target chordwise grid is the sole output of `cst.split_surface_t_values(num_output_points, point_density_beta)`:

```text
upper_x = 1 - upper_t    # trailing edge (1) to leading edge (0)
lower_x = lower_t        # leading edge (0) to trailing edge (1)
```

`upper_t` and `lower_t` are no longer applied to contour arc length. They define the fixed `x` coordinates shared by every processed airfoil and by `CSTDecoderLayer`. `point_density_beta = 1` gives uniform chordwise sampling; values above one concentrate points near the leading edge and values below one concentrate points near the trailing edge.

For each surface, linear interpolation evaluates the normalized source surface at its target coordinates:

```text
y_upper = linear_interpolate(source_upper_x, source_upper_y, upper_x)
y_lower = linear_interpolate(source_lower_x, source_lower_y, lower_x)
```

The output coordinate sequence is upper trailing edge to leading edge followed by lower leading edge to trailing edge, with the duplicate lower leading-edge point removed. Therefore every processed airfoil has an identical ordered `x` vector and its ordered `y` vector is a complete geometry representation. The GAN discriminator and aerodynamic surrogate may subsequently use a single `y` input channel, but only after this dataset representation is implemented and all dependent models are retrained.

`utils.normalize_airfoil_chord_coordinates` is the single implementation of this transform. It accepts one NumPy airfoil or one/batched PyTorch airfoils; the PyTorch path remains differentiable. File preprocessing supplies its validated leading-edge index. GAN-generated coordinates determine the minimum-x leading edge and are transformed into this local unit-chord frame before evaluation. The same normalized geometry is then used by both the surrogate model and XFoil, so their aerodynamic inputs describe one geometry in one pose; XFoil executes `NORM` as an idempotent safeguard and then `PANE` to redistribute only its internal analysis panels from the loaded spline. `PANE` never rewrites the processed or generated coordinate arrays.

`airfoil_preprocess.max_relative_thickness` in `config.yaml` defines the output thickness limit. `foildata/manage_foildata.py` owns the excluded-source-file list. Fixed-grid resampling, chord normalization, strict thickness filtering, and output writing operate on the same processed coordinate array. Changing the sampling representation requires rebuilding the processed airfoils, shared dataset, dataset split, coordinate normalizations, CST encodings, and all GAN and surrogate checkpoints.
