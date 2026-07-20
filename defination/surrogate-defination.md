# 气动代理模型定义

## 范围

`AerodynamicSurrogate` 根据翼型坐标和工况预测气动系数。输入为归一化翼型坐标与 `c = [alpha, Re]`，输出为归一化目标 `t = [CM, CL, CD]`。坐标经两层一维卷积提取特征，再与工况拼接并经全连接层回归。

卷积结构的唯一配置来源是 `surrogate_conv1_channels`、`surrogate_conv1_kernel`、`surrogate_conv2_channels`、`surrogate_conv2_kernel`、`surrogate_conv2_stride`。两个卷积核尺寸必须为正奇数；第二层卷积后的序列长度必须为正。

## 数据与划分

原始数据集由 `surrogate_dataset.data_path` 指定，供代理模型和 GAN 共用。样本含坐标 `x`、标签 `y = [alpha, Re, CL, CM]`、`cd` 与 `foil_id`。

`prepare_dataset.py` 只生成 `airfoil_group` 划分清单：同一 `foil_id` 的全部工况不会跨开发集、五折验证 fold 或独立测试集。清单保存到 `surrogate_dataset.split_path`，包含开发集、测试集和 `surrogate_cv_fold_count` 个 fold 索引。项目不支持按随机样本划分。

Optuna 仅在开发集内进行五折交叉验证。每个 trial 的每个 fold 用其余四折训练、当前折验证；测试集不参与调参、剪枝或模型选择。所有 fold 固定训练 `surrogate_cv_epochs` 个 epoch，并以最后一个 epoch 的验证加权 MSE 作为 fold 分数。

选定超参数后，最终模型从随机初始化开始，以完整开发集固定训练同样的 `surrogate_cv_epochs`。独立测试集仅由 `eval_surrogate.py` 使用。

## 归一化与损失

每个交叉验证 fold 的坐标、条件和目标归一化统计量只由该 fold 的训练索引计算。最终模型与 GAN 辅助损失使用完整开发集统计量，保存在 `surrogate_dataset.norm_path`，其中 `source_split` 必须为 `development`。

坐标的 x、y 分量各自采用 Min-Max 归一化；条件和目标采用 Z-score。最终归一化文件必须记录条件顺序 `[alpha, Re]` 与目标顺序 `[CM, CL, CD]`。

训练损失为归一化目标空间的加权 MSE：

`L = mean(w * (t_hat - t)^2)`

其中 `w = surrogate_target_loss_weights`，顺序固定为 `[CM, CL, CD]`。训练日志还记录反归一化后的逐目标 MAE。

## 训练与产物

`train_surrogate.py` 使用 Adam 与 `CyclicLR(mode='triangular2')`。训练张量在归一化后一次性放到目标设备；设备端索引完成打乱和 batch 切分。`surrogate_gradient_norm_interval` 仅控制梯度范数诊断采样，不影响反向传播或优化器更新频率。

最终模型保存至 `surrogate_dataset.best_model_path`，checkpoint 的 `selection_policy` 固定为 `fixed_final_epoch`，并记录 `training_epoch_count` 与归一化文件路径。`eval_surrogate.py` 只接受这种 checkpoint，并输出 `model/surrogate_test_metrics.yaml`、逐目标测试散点图和测试集 MAE、RMSE、R2。
