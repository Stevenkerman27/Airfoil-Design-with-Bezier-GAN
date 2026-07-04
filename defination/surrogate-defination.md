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
2. 坐标使用 Min-Max 归一化。
3. 输入工况只使用标签中的 `alpha, Re`，并单独计算均值和标准差。
4. 监督目标使用 `[CM, CL, CD]`：
   - `CL` 来自 `item["y"][2]`
   - `CM` 来自 `item["y"][4]`
   - `CD` 来自 `item["cd"]`
5. 目标 `[CM, CL, CD]` 单独计算均值和标准差，训练和评估时使用同一归一化参数。
6. 训练脚本保存代理模型归一化参数，作为推理和绘图的唯一来源。

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
5. 每 10 个 epoch 判断一次验证 loss 是否刷新最优模型；训练过程中最佳权重只保存在内存中。
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
