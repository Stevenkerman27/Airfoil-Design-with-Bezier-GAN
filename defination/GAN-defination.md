# CWGAN-GP 翼型设计定义

## 数据与条件

`train.py` 在 `surrogate_dataset.data_path` 指向的共用原始数据集上训练条件 WGAN-GP。训练仅使用五折划分清单的 `development_indices`；独立测试集不参与 GAN 参数更新或 GAN 归一化统计量。

条件固定为 `c = [alpha, Re, CL, CM]`，由 `cond_dim = 4` 约束。厚度不是条件输入，也不进入损失。`AirfoilDataset` 基于开发集覆盖写入：

- `model/cond_norm.pt`：四维条件的 Z-score 统计量；
- `model/coord_norm.pt`：翼型坐标 x、y 的 Min-Max 统计量。

生成器、训练辅助项和 GAN 评估均使用这两个文件。冻结代理模型及其归一化状态分别来自 `surrogate_dataset.best_model_path` 与 `surrogate_dataset.norm_path`。代理 checkpoint 必须为 `selection_policy = fixed_final_epoch`，归一化状态必须来自 `development`，条件和目标顺序分别为 `[alpha, Re]`、`[CM, CL, CD]`。

## 网络

`Generator(z, c)` 将高斯噪声和归一化条件输入 `gen_hid_layer` 个 `Linear + LeakyReLU(0.2)` 块，并输出 CST 曲线参数。

令 `N = cst.shape_coefficient_count`。GAN 每个表面使用 `N` 个形状系数，即 `N - 1` 次 Bernstein 多项式；`N` 必须至少为 2。每个生成样本输出 `2N + 4` 维参数：

```text
[A_upper[0:N], A_lower[0:N], delta_z_upper, delta_z_lower, N1, N2]
```

对物理弦坐标 `x in [0, 1]`，其中 `x=0` 为前缘，`x=1` 为后缘：

```text
C(x) = x^N1 * (1-x)^N2
S_surface(x) = sum(i=0..N-1, A_surface_i * B_i^(N-1)(x))
y_surface(x) = C(x) * S_surface(x) + x * delta_z_surface
```

`N1` 和 `N2` 是每个翼型独立生成的可学习参数，分别经 sigmoid 映射到 `cst.n1_range` 和 `cst.n2_range`，保证为正且数值稳定。`delta_z_upper`、`delta_z_lower` 保留有限尾缘厚度；前缘固定为 `(0, 0)`。CST 先在物理弦坐标计算，再依据 `model/coord_norm.pt` 转换为 GAN 坐标归一化空间。上下表面删除重复前缘点后得到 `num_output_points` 个有序坐标点：上后缘到前缘再到下后缘。

`Discriminator(x, c)` 将坐标重排为 `(batch, 2, point)`，经两层 `Conv1d + LeakyReLU(0.2)`、展平、条件拼接和全连接层输出不带 sigmoid 的 critic 分数。

## 损失与调度

判别器每个 batch 更新一次：

`L_D = -mean(D(x_real, c)) + mean(D(G(z, c), c)) + lambda_gp * GP`

`GP = mean((||grad_x_hat D(x_hat, c)||_2 - 1)^2)`，其中 `x_hat` 为真实与生成坐标的随机插值。

生成器对抗项为 `L_adv = -mean(D(G(z, c)))`。辅助项将生成坐标由 GAN 归一化空间转换至代理归一化空间，代理只接收 `[alpha, Re]`，并分别计算目标 `[CM, CL]` 的 MSE：

`L_surr = mean([w_CM * MSE_CM, w_CL * MSE_CL])`

`[w_CM, w_CL]` 由 `gan_surrogate_target_loss_weights` 提供。几何正则化采用论文的后缘端点交叉项。对第 `s` 个翼型，CST 上、下表面的后缘项分别为闭合曲线首尾端点的纵坐标 `y_upper_TE` 和 `y_lower_TE`，定义：

令 `x_1, ..., x_N` 为上表面输出采样中从后缘向弦内数的 `N = gan_trailing_edge_crossing_point_count` 个横坐标，重新在两表面共同的这些位置求值。`d_i = ReLU(y_lower(x_i) - y_upper(x_i))`，并令线性权重 `w_N = 0`、`w_1 = gan_trailing_edge_crossing_te_weight`，则：

`L_TE = mean_s(sum(i=1..N, w_i * d_i))`

当前 `N=3`。最后缘端点的权重是 `gan_trailing_edge_crossing_te_weight`，最靠内的第 `N` 组权重为 `0`，中间组线性插值。因此第三对点仅保留为采样边界，不贡献损失或梯度。该项只在相应组的下表面高于上表面时产生梯度；所有有效权重的组排序正确后损失和梯度均为零，不要求最小后缘厚度，也不在整条弦向上比较上下表面。

总生成器损失为 `L_G = a(epoch) * L_adv + s(epoch) * L_surr + L_TE`。几何项从第 0 个 epoch 起保持恒定权重；`gan_aux_start_epoch` 前气动辅助项为零，之后在 `gan_aux_ramp_epochs` 内将对抗权重过渡到 `gan_adv_loss_final_weight`，并将气动辅助权重升至 `gan_surrogate_loss_weight`。

## 训练、评估与产物

每个 batch 更新判别器；当 `i % n_critic == 0` 时更新生成器。两个优化器均为 `Adam(lr, betas=(0.0, 0.9), weight_decay=5e-5)`。训练不支持断点恢复，每次覆盖：

- `model/gan_training_metrics.csv`
- `model/gan_final.pt`
- `model/loss_curve.png`

`train.py` 写出的 GAN checkpoint 同时包含 `generator_state_dict` 和 `discriminator_state_dict`；`test_cgan.py` 拒绝裸生成器 state dict。CST 生成器与此前 Bezier 生成器 checkpoint 不兼容，需重新训练。

`eval_cgan.py` 仅从独立测试集抽样条件并用 XFoil 评估。每个条件生成 `k_samples` 个翼型，评分为：

`weighted_error = w_CM * |CM_xfoil - CM_target| + w_CL * |CL_xfoil - CL_target|`

每个条件的热图值为有效样本的平均绝对误差加 `eval_var_weight * variance`。`test_cgan.py` 接收用户给定的 `[alpha, Re, CL, CM]`，报告 XFoil、冻结代理与目标之间的误差。测试脚本对非负整数目标迎角显式启用 XFoil continuation：从 `0°` 开始以 `1°` 步进至目标迎角，每个迎角最多迭代 50 次，只采用目标迎角的完整 `CL/CD/CM`；总进程超时按迎角计算次数乘以单点超时预算。负迎角或非整数迎角立即报错。其他 `run_xfoil_single` 调用默认仍只计算指定迎角。

`--development-samples` 默认值为 5，脚本使用 `surrogate_seed` 从划分清单的 `development_indices` 中无放回抽取 N 个真实四维条件，每个条件生成 `NUM_GENERATE` 个翼型；N 大于 0 时该模式覆盖 `--labels`，输出标签附加 `DEV###` 以区分条件。显式指定 `--development-samples 0` 时才使用 `--labels`。

## Critic sensitivity diagnostic

`visualize_discriminator_sensitivity.py` differentiates the scalar critic score with respect to every GAN-normalized input coordinate using `autograd.grad(D(coords, c).sum(), coords)`. It reports both this direct discriminator-input gradient and its physical-coordinate form, obtained by dividing `dD/dx_normalized` and `dD/dy_normalized` by the corresponding ranges in `model/coord_norm.pt`. The script samples five development-set real airfoils by default and can instead generate samples for a supplied `[alpha, Re, CL, CM]` condition. Each selected airfoil produces a CSV and a figure showing physical geometry, physical-gradient direction, and both gradient coordinate systems. `mean.csv` and `mean.png` aggregate selected airfoils by ordered point index with equal sample weights; they report both the norm of the mean gradient and the mean of per-sample L2 sensitivity, which are distinct quantities. Normalized and physical derivatives must not be interpreted as interchangeable.
