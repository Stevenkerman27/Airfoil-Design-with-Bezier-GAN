# 代理模型五折交叉验证

## 目标

代理模型采用“独立测试集 + 开发集五折交叉验证”，在不使用测试集调参的前提下，提高超参数选择的稳定性。

## 数据划分

1. 先从原始数据中按 `foil_id` 分组划出独立测试集，比例由 `surrogate_test_ratio` 配置。
2. 剩余数据为开发集；开发集继续按 `foil_id` 分组划分为 `surrogate_cv_fold_count=5` 个 fold。
3. `airfoil_group` 策略以 `foil_id` 为不可分割的组：同一翼型不得跨训练 fold、验证 fold 或独立测试集。
4. 项目不再支持按随机样本划分的代理模型协议。
5. 划分清单必须记录外层测试索引和全部开发集 fold 索引，并校验索引覆盖、互斥、非空和分组隔离。

## 训练与超参数优化

每个 Optuna trial 依次训练五个 fold。每个 fold 使用其余四折训练、当前一折验证；归一化统计量只由该 fold 的训练样本计算。

所有 fold 固定训练 `surrogate_cv_epochs` 个 epoch：

- 不记录或恢复最低验证损失时的权重。
- 完成固定轮数后，使用最后一轮权重计算该 fold 的验证损失。
- trial 目标为五个最终验证加权 MSE 的算术平均值。
- 记录五折损失的均值和标准差，以及反归一化后的逐目标指标，供诊断使用。
- 测试集不得参与 trial、剪枝、epoch 选择或超参数选择。

## 最终模型与测试

选定超参数后，从随机初始化重新创建一个最终模型，以整个开发集训练固定 `surrogate_cv_epochs` 个 epoch。最终模型仅保存最后一轮权重，并以开发集统计量完成归一化。

最终模型供 GAN 的冻结代理损失使用。训练完成后只对独立测试集进行评估，输出总体及 `[CM, CL, CD]` 的 MAE、RMSE、R2 和预测散点图；测试结果不得反向用于修改模型或配置。

## 配置与约束

交叉验证相关参数集中在 `config.yaml`，包括测试集比例、fold 数、随机种子、固定训练 epoch 数，以及 `surrogate_dataset` 的数据划分、归一化和最终模型路径。`surrogate_cv_fold_count` 必须大于 1；翼型组数必须足以同时填充测试集与每个开发集 fold。

## 训练性能与日志

- 代理模型完成归一化后，将完整的 `coords`、`conditions` 和 `targets` 连续张量一次性放入 `device`。训练时在设备端打乱索引并切分 batch；验证和测试按设备端索引顺序切分。代理模型不使用 DataLoader、worker 或 CPU-GPU 的逐 batch 传输。
- 训练损失、MAE 和逐目标误差在 GPU 上按 batch 累计，仅在每个 epoch 结束后转回 CPU 写入指标，避免逐 batch 同步。
- `surrogate_gradient_norm_interval` 只控制梯度范数诊断的采样频率，不改变损失、反向传播、优化器更新或学习率调度。每个 epoch 采样首个 batch，并在后续每满一个间隔的 batch 采样一次；CSV 中的梯度范数均值和最大值基于这些采样值。
