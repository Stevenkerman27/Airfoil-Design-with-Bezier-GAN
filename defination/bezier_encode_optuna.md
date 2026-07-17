# Bezier Encoding Optuna

本文档记录翼型 Bezier 编码超参数优化流程。该流程只优化 `.dat` 翼型到有理 Bezier 曲线的重建超参数，不修改 CWGAN-GP 的 Generator、Discriminator 或训练流程。

当前编码支持两种曲线模式：

- `single`：使用一条有理 Bezier 曲线从尾缘出发，经前缘回到尾缘。
- `split_surface`：使用上下表面两条有理 Bezier 曲线，分别固定首尾端点。

默认使用 `split_surface`，该模式只影响编码和 Optuna 重建评估，当前不修改 GAN 的 `Generator` 或 `BezierDecoderLayer`。

## 目标

优化 `foildata/processed_foil` 中全部翼型的 Bezier 重建精度，并在重建误差接近时偏向更少控制点。

目标函数定义为：

```text
objective = mean_mae + control_point_penalty_weight * total_control_points
```

其中：

- `mean_mae` 是所有参与优化翼型在归一化坐标系中的平均重建 MAE。
- `total_control_points` 是上下两条 Bezier 曲线的总控制点数量。
- `bezier_encode.surface_control_points` 是每一条表面曲线的控制点数量。
- `control_point_penalty_weight` 由 `config.yaml` 中 `bezier_encode_optuna.control_point_penalty_weight` 统一配置。

前缘误差目前只作为监控指标保存到 Optuna trial attributes，不进入目标函数。
内层 Bezier 拟合 loss 使用 cosine 前缘加权 MAE，前缘权重强调前缘区域；外层 Optuna 目标函数使用未加权 `mean_mae`。

## 配置来源

所有参数以 `config.yaml` 为单一来源：

- `bezier_encode`：单个翼型 Bezier 拟合的训练参数、输出路径和可视化路径。
- `bezier_encode_optuna`：Optuna study、数据目录、控制点数量惩罚和搜索空间。
- `bezier_encode.batch_size`：Optuna 评估时一次并行拟合的翼型数量。
- `bezier_encode.curve_mode`：Bezier 编码曲线模式，可选 `single` 或 `split_surface`。

当前搜索空间：

```yaml
bezier_encode_optuna:
  search_space:
    bezier_encode.surface_control_points: [4, 15]
    bezier_encode.lr: [1e-3, 5e-2, log]
    bezier_encode.iterations: [500, 750, 1000]
    bezier_encode.weight_reg: [1e-5, 1e-2, log]
    bezier_encode.length_penalty: [1e-4, 1.0, log]
    bezier_encode.leading_edge_window: [3, 15]
    bezier_encode.leading_edge_weight_amplitude: [0.0, 5.0, linear]
```

`point_density_beta` 使用 `config.yaml` 顶层固定值，不参与 Optuna 搜索。

## 双曲线拆分模式

`split_surface` 模式按每个翼型自己的最小 `x` 坐标点识别前缘，因此可以处理数据集中前缘索引为 `48`、`49`、`50` 或 `51` 的样本。

点序假设为：

```text
upper: TE_upper -> LE
lower: LE -> TE_lower
```

固定端点为归一化后的 `.dat` 实际点：

- 上表面：首点固定为上尾缘，末点固定为前缘。
- 下表面：首点固定为前缘，末点固定为下尾缘。

`bezier_encode.surface_control_points` 表示每一条表面曲线的控制点数量。分配规则为：

```text
upper_control_points = surface_control_points
lower_control_points = surface_control_points
total_control_points = 2 * surface_control_points
```

因此 Optuna 不再搜索奇数总控制点。`split_surface` 至少需要 `surface_control_points` 为 `2`，以保证每条曲线至少有两个固定端点。

该模式的编码输出中，`control_points` 和 `weights` 为字典：

```text
control_points.upper: (B, N_upper, 2)
control_points.lower: (B, N_lower, 2)
weights.upper: (B, N_upper)
weights.lower: (B, N_lower)
```

当前 GAN 仍使用单条曲线结构，不能直接把 `split_surface` 输出当作旧 Generator 的参数格式使用。等重建精度验证后，再决定是否同步改 GAN 的生成器输出结构。

## 运行方式

先进行小样本 smoke test：

```powershell
D:\Software\anaconda\envs\myml\python.exe optimize_encode_dat.py --n-trials 1 --max-airfoils 3
```

确认流程可运行后执行完整优化：

```powershell
D:\Software\anaconda\envs\myml\python.exe optimize_encode_dat.py
```

如果需要限制调试规模：

```powershell
D:\Software\anaconda\envs\myml\python.exe optimize_encode_dat.py --n-trials 5 --max-airfoils 50
```

PowerShell 5.1 中不要使用 `&&` 或 `||` 串联命令。

## 输出

最佳结果保存到：

```text
model/bezier_encode_optuna_best.yaml
```

文件包含：

- `best_value`
- `best_params`
- `best_trial_number`
- `best_metrics.mean_mae`
- `best_metrics.mean_mse`
- `best_metrics.mean_leading_edge_mae`
- `best_metrics.mean_leading_edge_mse`
- `best_metrics.max_point_error`
- `best_metrics.surface_control_points`
- `best_metrics.total_control_points`

单翼型编码仍由 `encode_dat.py` 执行，输出路径由 `bezier_encode.output_path` 和 `bezier_encode.plot_path` 控制。

## Batch 化评估

`optimize_encode_dat.py` 会按 `bezier_encode.batch_size` 将翼型切分成多个 batch，并在每个 batch 内同时优化每个翼型独立的控制点和权重。

张量形状为：

```text
target_points: (B, M, 2)
single.trainable_control_points: (B, N - 2, 2)
single.weights: (B, N)
split.upper_trainable_control_points: (B, N_upper - 2, 2)
split.lower_trainable_control_points: (B, N_lower - 2, 2)
split.upper_weights: (B, N_upper)
split.lower_weights: (B, N_lower)
curve: (B, M, 2)
```

其中 `B` 是 batch size，`M` 是输出点数，`N` 是单曲线控制点数。Batch 化不共享控制点，也不改变重建目标，只减少 Python 循环开销并提高 GPU 利用率。

如果显存不足，降低：

```yaml
bezier_encode:
  batch_size: 32
```

## 前缘 Cosine 加权

内层 Adam 拟合使用 cosine 前缘权重计算 MAE：

```text
weight(d) = 1 + A * 0.5 * (1 + cos(pi * d / W)),  d <= W
weight(d) = 1,                                      d > W
```

其中：

- `d` 是点到前缘点的索引距离。
- `W` 是 `bezier_encode.leading_edge_window`，同一个窗口也用于最终前缘 MAE/MSE 统计。
- `A` 是 `bezier_encode.leading_edge_weight_amplitude`。

前缘点最大权重为 `1 + A`，窗口边界处平滑回到 `1`。该权重只影响每个翼型的内层拟合 loss；最终记录的 `mean_mae`、`mean_mse`、`mean_leading_edge_mae`、`mean_leading_edge_mse` 使用未加权重建误差。
