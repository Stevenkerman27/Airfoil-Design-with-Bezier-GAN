# 神经网络技术路线

## 目标

新增一个气动代理模型，用于从翼型坐标和工况预测气动系数：

- 输入：归一化翼型坐标 `coords` 与归一化工况 `alpha, Re`
- 输出：归一化目标 `[CM, CL, CD]`
- 训练后在验证集上反归一化并绘制 `CL, CD, CM` 的真实值-预测值散点图和 45 度参考线

## 已对齐决策

1. 采用输入方案 B：`翼型坐标 + alpha + Re -> CM, CL, CD`。
2. 数据集按样本级随机划分，比例为训练集:验证集:测试集 = 8:1:1。
3. 模型定义写入 `model.py`。
4. 新建独立训练脚本，不改动现有 CWGAN-GP 训练入口。

## 数据处理路线

1. 从 `model/airfoil_dataset.pt` 读取原始数据。
2. 使用固定随机种子按样本级随机划分训练集、验证集、测试集，比例为 8:1:1。
3. 坐标使用训练集统计量做 Min-Max 归一化。
4. 输入工况只使用标签中的 `alpha, Re`，并使用训练集统计量计算均值和标准差。
5. 监督目标使用 `[CM, CL, CD]`：
   - `CM` 来自 `item["y"][4]`
   - `CL` 来自 `item["y"][2]`
   - `CD` 来自 `item["cd"]`
6. 目标 `[CM, CL, CD]` 使用训练集统计量计算均值和标准差，训练、验证、测试和推理时使用同一归一化参数。
7. 训练脚本保存代理模型归一化参数，作为推理、评估和绘图的唯一来源。

## 归一化定义

归一化统计量只从训练集计算，禁止使用验证集和测试集参与统计量估计，避免数据泄漏。验证集和测试集只应用训练集统计量。

设训练集样本索引集合为 `I_train`，目标顺序固定为 `[CM, CL, CD]`。

### 坐标归一化

翼型坐标原始值为 `(x, y)`。训练集上的坐标范围定义为：

`x_min = min_{i in I_train, p} x_{i,p}`

`x_max = max_{i in I_train, p} x_{i,p}`

`y_min = min_{i in I_train, p} y_{i,p}`

`y_max = max_{i in I_train, p} y_{i,p}`

所有 train/val/test 样本统一使用：

`x_norm = (x - x_min) / (x_max - x_min + 1e-8)`

`y_norm = (y - y_min) / (y_max - y_min + 1e-8)`

### 工况归一化

工况输入为：

`c = [alpha, Re]`

训练集统计量为：

`condition_mean = mean_{i in I_train}(c_i)`

`condition_std = std_{i in I_train}(c_i) + 1e-8`

所有 train/val/test 样本统一使用：

`c_norm = (c - condition_mean) / condition_std`

### 目标归一化

监督目标为：

`t = [CM, CL, CD]`

训练集统计量为：

`target_mean = mean_{i in I_train}(t_i)`

`target_std = std_{i in I_train}(t_i) + 1e-8`

所有 train/val/test 样本统一使用：

`t_norm = (t - target_mean) / target_std`

模型训练时预测的是归一化目标空间中的 `t_norm`。评估 MAE、RMSE、R2 和散点图时，需要反归一化回真实气动系数尺度：

`t = t_norm * target_std + target_mean`

### 归一化参数保存

`model/surrogate_norm.pt` 保存 train-only 归一化参数，并记录：

- `source_split: train`
- `split_seed`
- `split_ratio`
- `split_counts`
- 坐标范围 `x_min, x_max, y_min, y_max`
- 工况 `condition_mean, condition_std`
- 目标 `target_mean, target_std`

## 模型路线

1. 在 `model.py` 新增 `AerodynamicSurrogate`。
2. 坐标特征提取架构参考现有 `Discriminator`：
   - `Conv1d(2 -> disc_conv_channels)`
   - `Conv1d(disc_conv_channels -> disc_conv2_channels)`
   - 展平后拼接归一化工况 `[alpha, Re]`
   - 多层全连接层
3. 输出层维度为 3，对应 `[CM, CL, CD]`。

## 损失函数定义

代理模型使用监督回归加权 MSE 损失。对一个 batch，模型预测值为 `y_hat`，归一化真实目标为 `y`，目标维度数量为 `K = 3`，batch size 为 `B`，标签损失权重为 `w`。

损失函数定义为：

`L = (1 / (B * K)) * sum_{b=1..B} sum_{k=1..K} w[k] * (y_hat[b, k] - y[b, k])^2`

其中目标顺序固定为 `[CM, CL, CD]`，权重配置项 `surrogate_target_loss_weights` 也使用相同顺序。训练损失和验证损失都在归一化目标空间中计算；训练和验证 MAE 为反归一化到真实气动系数后的平均绝对误差。

## 训练路线

1. 新建 `train_surrogate.py`。
2. 使用固定随机种子进行样本级 8:1:1 划分。
3. 损失函数采用可配置标签权重的加权 MSE，优化器采用 `Adam`。
4. 每个 epoch 记录：
   - 训练 loss
   - 验证 loss
   - 训练 MAE
   - 验证 MAE
5. 每个 epoch 判断一次验证 loss 是否刷新最优模型；训练过程中最佳权重只保存在内存中。
6. 训练结束后，将内存中的最佳模型保存到 `model/surrogate_best.pt`。
7. 最终用最优模型在验证集绘图。

## 绘图输出

训练过程输出：

- `model/surrogate_loss.png`：训练集和验证集 loss 曲线
- `model/surrogate_error.png`：训练集和验证集 MAE 曲线

验证集预测输出：

- `model/surrogate_val_cl.png`
- `model/surrogate_val_cd.png`
- `model/surrogate_val_cm.png`

每张验证图横轴为实际值，纵轴为预测值，并包含 45 度完全精确参考线。

## 验证路线

1. 运行现有相关单元测试，确认未破坏判别器和数据集行为。
2. 运行代理模型训练脚本，至少完成一次端到端训练和绘图。
3. 如训练耗时过长，先执行较小 epoch 的冒烟测试，确认模型、数据、绘图和保存流程正确。
