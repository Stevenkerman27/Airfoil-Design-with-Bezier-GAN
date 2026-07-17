# CWGAN-GP 翼型设计定义

## 数据与划分

原始数据集为 `model/airfoil_dataset.pt`。每个样本保存展平坐标 `x`、条件 `y = [alpha, Re, CL, CM]`、阻力系数 `cd` 与 `foil_id`。

`surrogate_dataset_name` 选择随机样本或按翼型分组协议。GAN、气动代理都只使用所选清单的 `train` 索引；验证和测试索引不参与 GAN 参数更新或 GAN 归一化统计量计算。

GAN 条件采用逐维 Z-score，坐标的 x、y 分量采用训练子集的 Min-Max 归一化，统计量写入 `model/cond_norm.pt` 与 `model/coord_norm.pt`。

## 网络

- `Generator(z, c)`：输入高斯噪声和 4 维条件，经 MLP 输出有理 Bezier 控制点及正权重；首尾控制点固定为归一化尾缘，输出 100 个翼型坐标点。
- `Discriminator(x, c)`：两层 Conv1d 提取坐标序列特征，与 4 维条件拼接后输出一个 WGAN critic 分数，不使用 sigmoid。
- `AerodynamicSurrogate(x, [alpha, Re])`：冻结代理网络，输出 `[CM, CL, CD]`；其模型和归一化文件由选中数据集条目的 `best_model_path` 与 `norm_path` 指定。

## 损失

Critic 损失：

`L_D = -mean(D(x_real, c)) + mean(D(G(z,c), c)) + lambda_gp * GP`

其中 `GP = mean((||grad_x_hat D(x_hat,c)||_2 - 1)^2)`，`x_hat` 是真实和生成坐标的随机线性插值。

生成器对抗项：

`L_adv = -mean(D(G(z,c), c))`

辅助气动项先将生成坐标变换到代理归一化空间，分别计算归一化 MSE：

`L_surr = mean(10 * MSE_CM + 1 * MSE_CL)`

权重由 `gan_surrogate_target_loss_weights: [10.0, 1.0]` 定义，顺序固定为 `[CM, CL]`。总生成器损失为：

`L_G = w_adv * L_adv + w_surr * L_surr`

`w_surr` 在 `gan_aux_start_epoch` 后按 `gan_aux_ramp_epochs` 线性增加。厚度不再是条件、辅助损失或优化目标。

## 评估

`eval_cgan.py --split val` 用验证条件评估 checkpoint；`--split test` 仅用于最终测试。两者都通过 XFoil 计算实际 `CM` 与 `CL`，并报告：

`weighted_error = 10 * abs(CM_pred - CM_target) + abs(CL_pred - CL_target)`

Critic loss 仅用于训练诊断，不用于选择气动性能最佳的 checkpoint。
