# CWGAN-GP 翼型设计定义

## 范围

`train.py` 训练条件 WGAN-GP，生成翼型坐标。条件固定为 `c = [alpha, Re, CL, CM]`；厚度不是条件或损失项。训练数据来自 `surrogate_dataset.data_path`，并严格使用五折清单的 `development_indices`。独立测试集不参与 GAN 参数更新或 GAN 归一化统计量。

## 数据与归一化

原始样本包含展平坐标 `x`、标签 `y = [alpha, Re, CL, CM]`、`cd` 和 `foil_id`。`AirfoilDataset` 仅以开发集计算并保存：

- `model/cond_norm.pt`：四维条件的 Z-score 均值和标准差。
- `model/coord_norm.pt`：坐标 x、y 分量的 Min-Max 范围。

每次运行 `train.py` 都会覆盖这两个文件。生成器、训练辅助项和 `eval_cgan.py` 都从这两个固定路径读取统计量。

冻结代理来自 `surrogate_dataset.best_model_path`，归一化文件来自 `surrogate_dataset.norm_path`。仅当 checkpoint 的 `selection_policy` 为 `fixed_final_epoch`，且代理归一化文件的 `source_split` 为 `development`、条件顺序为 `[alpha, Re]`、目标顺序为 `[CM, CL, CD]` 时，才可启用辅助损失。

## 网络

`Generator(z, c)`：将 `noise_dimension` 维高斯噪声和四维归一化条件拼接，经 `gen_hid_layer` 个 `Linear + LeakyReLU(0.2)` 块输出 `num_control_points` 个二维控制点和权重。权重经过 `softplus` 保证为正；首尾控制点固定为归一化尾缘。`BezierDecoderLayer` 用有理 Bezier 曲线采样 `num_output_points` 个二维坐标并展平输出。

`Discriminator(x, c)`：将坐标重排为 `(batch, 2, point)`，依次通过两层带同尺寸 padding 的 `Conv1d + LeakyReLU(0.2)`，展平后拼接条件，经 `dis_hid_layer - 1` 个全连接隐藏块，输出一个不含 sigmoid 的 critic 分数。

## 损失与调度

判别器每个 batch 更新一次：

`L_D = -mean(D(x_real, c)) + mean(D(G(z, c), c)) + lambda_gp * GP`

`GP = mean((||grad_x_hat D(x_hat, c)||_2 - 1)^2)`，其中 `x_hat` 是真实与生成坐标的随机插值。

生成器对抗项为 `L_adv = -mean(D(G(z, c), c))`。辅助项先把生成坐标从 GAN 归一化空间反归一化，再变换到代理归一化空间；条件仅取 `[alpha, Re]`。代理预测与目标的归一化 `[CM, CL]` 分别计算 batch MSE：

`L_surr = mean([w_CM * MSE_CM, w_CL * MSE_CL])`

其中 `[w_CM, w_CL] = gan_surrogate_target_loss_weights`。总生成器损失为：

`L_G = a(epoch) * L_adv + s(epoch) * L_surr`

当 epoch 小于 `gan_aux_start_epoch` 时，`a = 1`、`s = 0`。之后的 `gan_aux_ramp_epochs` 内，进度为 `p = (epoch - start + 1) / ramp`，并取 `a = 1 + p * (gan_adv_loss_final_weight - 1)`、`s = p * gan_surrogate_loss_weight`；调度结束后 `p = 1`。

## 训练与产物

每个训练 batch 都更新判别器；当 batch 索引满足 `i % n_critic == 0` 时更新生成器，因此每个 epoch 的第一个 batch 会更新生成器。两个优化器均为 `Adam(lr, betas=(0.0, 0.9), weight_decay=5e-5)`。DataLoader 使用 `shuffle=True` 和 `drop_last=True`。

训练不支持中断后恢复，每次从随机初始化开始，并覆盖：

- `model/gan_training_metrics.csv`：critic、生成器、辅助项、梯度范数和调度权重。
- `model/gan_final.pt`：仅含 `generator_state_dict` 和 `discriminator_state_dict`。
- `model/loss_curve.png`：训练指标图。

GAN checkpoint 不保存数据集清单、归一化统计量或配置副本。加载旧 checkpoint 时，调用方必须自行保证当前固定归一化文件和网络配置与训练时一致。

## 评估与生成

`eval_cgan.py` 仅接受 `--split test`。它以 `surrogate_seed` 确定性地从独立测试集抽取条件，每个条件生成 `k_samples` 个翼型，调用 XFoil，并按：

`weighted_error = w_CM * |CM_xfoil - CM_target| + w_CL * |CL_xfoil - CL_target|`

评分。每个条件的热图值为该条件有效样本的平均绝对误差加 `eval_var_weight * 方差`。全局最小的 `top_m` 个有效结果写入 `foildata/gen`。`test_cgan.py` 则接受用户给出的 `[alpha, Re, CL, CM]`，同时报告 XFoil 与冻结代理的误差。
