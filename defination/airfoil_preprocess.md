# Airfoil Preprocessing

Processed airfoils use the same local chord frame as XFoil `NORM`: the minimum-x leading edge is `(0, 0)`, and the midpoint of the two trailing-edge points is `(1, 0)`.

`utils.normalize_airfoil_chord_coordinates` is the single implementation of this transform. It accepts one NumPy airfoil or one/batched PyTorch airfoils; the PyTorch path remains differentiable. File preprocessing supplies its validated leading-edge index. GAN training and generation evaluation determine the minimum-x leading edge, transform a coordinate copy before surrogate normalization, and leave the generated/XFoil coordinates unchanged.

`airfoil_preprocess.max_relative_thickness` in `config.yaml` defines the output thickness limit. `foildata/manage_foildata.py` owns the excluded-source-file list. Resampling, chord normalization, strict thickness filtering, and output writing operate on the same processed coordinate array.
