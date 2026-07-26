# GAN 探索与交替微调

## 范围与职责

`gan_conditions.py` 是 GAN 条件向量顺序的单一来源：`[alpha, Re, CL, CM]`。`gan_exploration.py` 定义条件分布、条件扩展、生成器与代理模型的联合推理以及潜变量优化。`explore_gan_latent.py` 用这些能力执行离线探索和 XFoil 诊断。`alternating_finetuning.py` 则以相同的条件扩展逻辑执行 GAN 与代理模型的交替微调。

原始独立测试集只用于代理模型报告，绝不用于条件统计、采样、生成训练、判别器训练或代理模型微调。开发集是原始数据参与这些流程的唯一来源。

GAN 探索与第 0 轮交替微调均以 `model/surrogate_airfoil_group_best.pt` 作为初始代理模型；它由 `train_surrogate.py` 从开发集训练产生，并与 `model/surrogate_airfoil_group_norm.pt` 配套。后续交替轮次只读取上一轮保存的代理 checkpoint，不再回退到初始模型。

## 条件分布与扩展

条件按精确的 `(alpha, Re)` 分层。每层收集开发集及历轮 XFoil 成功生成记录的实际 `[CL, CM]`，共同估计层内均值 `mu` 与经验协方差 `Sigma`。`CL` 和 `CM` 必须作为二维联合随机变量处理，禁止分别独立采样或任意拼接，以保持同一工况下的相关关系。

扩展样本在白化空间内均匀采样马氏椭圆：先取随机方向 `u` 和半径 `r = R * sqrt(U)`，再映射为 `mu + L(r * u)`，其中 `L L^T = Sigma`，`R` 为马氏半径上限。故每个扩展点都满足：

```text
(x - mu)^T Sigma^(-1) (x - mu) <= R^2
```

采样记录保留 `source`、源数据集索引和实际马氏半径，便于报告可追溯。报告同时给出全局 `CL/CM` 相关性和去除各 `(alpha, Re)` 层均值后的层内相关性；只有后者可解释为同一工况下的统计关系。

独立探索使用 `gan_exploration.empirical_condition_fraction`，它表示总条件数中直接复用经验条件的比例，其余为协方差扩展条件。交替微调逐轮使用 `alternating_finetuning.existing_to_sampled_ratios`；例如 `[7, 3]` 表示该轮总条件中约 70% 直接来自当前已知条件池、约 30% 为协方差扩展条件。`mahalanobis_radii` 与该比例列表一一对应，列表长度定义总轮数。

`condition_count` 与 `condition_count_per_round` 均是一次运行或一轮中的总条件数，不是每个 `(alpha, Re)` 层的数量。每个条件随后生成 `noise_samples_per_condition` 个翼型；例如 `24 * 12 = 288` 是一轮最多的 XFoil 请求数。

## 潜空间探索与 XFoil 诊断

随机诊断固定一个扩展条件，以 `z ~ N(0, I)` 生成多组翼型。生成器输出先经现有翼型规范化流程变为单位弦长坐标；代理模型只接收其自身训练时使用的坐标和工况归一化形式。代理模型预测仅用于筛选和误差对照，不能替代 XFoil 标签。

低阻模式从多个随机 `z_initial` 出发，冻结 GAN 和代理模型，最小化：

```text
L_latent = mean(CD_pred)
         + w_CL * mean(max(|CL_pred - CL_target| - tol_CL, 0)^2)
         + w_CM * mean(max(|CM_pred - CM_target| - tol_CM, 0)^2)
```

`cl_tolerance` 和 `cm_tolerance` 是允许的目标跟踪误差；误差在容差以内不产生罚项。`cl_penalty_weight` 和 `cm_penalty_weight` 决定超过容差后保留 `CL/CM` 的强度。每一步优化后将 `z - z_initial` 投影回半径为 `latent_trust_radius` 的球内，限制搜索离开训练潜分布的距离。优化结果只是更值得运行 XFoil 的候选，绝不直接作为真实气动样本。

`explore_gan_latent.py` 对每个候选运行 XFoil，并报告逐条件和总体的 `CM/CL/CD` 均值、标准差、范围、收敛率以及代理模型相对 XFoil 的误差。随机模式与低阻优化模式分别输出，避免把噪声敏感性与优化效果混在一起。

## XFoil 成功记录

生成翼型进入训练数据前，必须先变换到统一的局部单位弦长坐标系。记录、代理模型和 XFoil 均使用这份规范化后的同一几何；XFoil 内部的 `NORM` 作为幂等保护保留。只有 XFoil 返回的 `CM`、`CL`、`CD` 均存在且有限时，记录才是成功记录；任何一个缺失、非有限、几何预处理失败或 XFoil 未收敛都不能加入数据集。

成功记录保存生成坐标、请求条件、噪声、实际 XFoil `[CM, CL, CD]` 和相应状态。后续流程中的 GAN 条件必须使用实际 XFoil `CL/CM`，而不是当初请求的 `CL_target/CM_target`。历轮成功记录组成累计生成数据集，并扩展下一轮的条件协方差统计、代理模型生成样本池和判别器真样本池。

## 交替微调外循环

`alternating_finetuning.py` 对第 `k` 轮按以下顺序执行：

```text
冻结 G_k
  -> 从当前已知条件池采样并扩展条件
  -> 每个条件生成多个翼型并运行 XFoil
  -> 将本轮 XFoil 成功记录追加到累计集
  -> 固定 epoch 微调 S_k，得到 S_(k+1)
冻结 S_(k+1)
  -> 固定 epoch 训练 G_k 与判别器，得到 G_(k+1)
```

每轮固定训练 `surrogate_epochs_per_round` 和 `gan_epochs_per_round`，不做 checkpoint 选择；保存固定 epoch 后的最终模型。代理模型和 GAN 的每个 epoch 都打印平均损失，并写入该轮 YAML 的 `epoch_metrics` 列表；`final_epoch_losses` 保留为该列表最后一项，供既有读取逻辑使用。若一轮没有 XFoil 成功样本，代理模型无法获得本轮生成监督，该轮应失败而不是静默跳过。

### 代理模型微调

代理模型的输出误差始终使用全局 `surrogate_target_loss_weights` 所定义的同一套加权 MSE，输出顺序为 `[CM, CL, CD]`。该权重决定三个物理量的相对重要性，基础训练与交替微调共用，不能在两个阶段分别定义。

每个微调 step 同时抽取两批数据：一批来自累计 XFoil 成功生成记录，另一批来自原始开发集。生成批按 `surrogate_historical_to_new_ratio` 混合历史成功记录与本轮新增成功记录；原始批仅从开发集随机回放。每 epoch 的 step 数为两种样本覆盖所需步数的较大值：

```text
max(ceil(generated_success_count / surrogate_generated_batch_size),
    ceil(development_count / surrogate_original_replay_batch_size))
```

总微调损失为：

```text
L_surrogate = lambda_generated * L_generated
            + lambda_original * L_original
            + lambda_anchor * sum(||theta - theta_baseline||^2)
```

三个 `surrogate_lambda_*` 是数据来源和参数约束的权重，不是 `CM/CL/CD` 的输出权重。`L_original` 防止模型遗忘原始数据分布；`L_anchor` 可选地限制参数偏离本轮开始时的代理模型。原始独立测试集只在本轮开始与结束时评估。

### GAN 与判别器微调

判别器真样本池由原始开发集和累计 XFoil 成功生成集构成，并由 `gan_real_original_to_generated_ratio` 控制二者批内比例。真实生成记录的条件是其 XFoil 实测 `[alpha, Re, CL, CM]`。

对抗训练的假样本必须使用同一真样本池抽取的实测条件：

```text
z ~ N(0, I),  c_real ~ real_condition_pool,  x_fake = G(z, c_real)
```

因此判别器不能将“条件本身是否脱离经验分布”作为区分真假的捷径。协方差扩展条件禁止直接作为判别器的假条件。

扩展条件只用于生成器的气动辅助损失：冻结的代理模型预测生成翼型，生成器追踪扩展条件中的 `CM/CL` 目标。其内部目标权重仍来自 `gan_surrogate_target_loss_weights`，顺序为 `[CM, CL]`；交替阶段的总生成器损失为：

```text
L_G = gan_adversarial_weight * L_adversarial
    + gan_aerodynamic_weight * L_aerodynamic
    + L_TE
```

`gan_aerodynamic_weight` 是交替微调阶段的总辅助损失强度；它与基础 GAN 训练的随 epoch 调度参数分开，因两者的训练阶段和调度策略不同。`L_TE` 与基础 GAN 训练共用同一权威定义：在后缘起的 `gan_trailing_edge_crossing_point_count` 个共同弦向位置累加 `ReLU(y_lower - y_upper)`，权重由最靠内组的 `0` 线性升至端点的 `gan_trailing_edge_crossing_te_weight`。零权重最靠内组只作为采样边界，不产生梯度。该项仅约束后缘区域，不比较全弦最大穿越深度。交替训练同时对经验条件的对抗生成批次和扩展条件的气动生成批次计算该项并取平均。WGAN-GP 判别器仍使用 `lambda_gp` 梯度罚项。

## 产物与报告

累计生成数据集及每轮 GAN/代理模型 checkpoint 位于 `model/`，包括 `model/alternating_generated_dataset.pt` 和 `model/alternating_checkpoints/`。条件统计、每轮汇总 YAML 和系数散点图位于 `reports/alternating/`；探索输出位于 `reports/exploration/`。所有 YAML 报告在顶层写入 `generated_at`；CSV 指标行包含同名列；PNG 报告通过内嵌 `Creation Time` 元数据保存时间，文件名保持稳定并允许覆盖。

交替微调的系数散点图会从原始开发集、历史成功生成集和本轮新增成功生成集的实际数据中收集全部 `(alpha, Re)` 分层，不受静态绘图子集限制。迎角沿水平方向排列，雷诺数沿垂直方向排列。图继续以现有 `Cd` 色图编码阻力、以既有 marker 区分原始、历史生成和本轮新增成功样本，确保每个成功生成点都落入某个面板；每个面板标注本轮前已有点总数及本轮新增成功点数。

## 交替微调的均衡 XFoil 采样

交替微调以精确的 `(alpha, Re)` 分层。每层必须收集 `successful_samples_per_operating_condition` 个严格收敛的 XFoil 新样本；`max_xfoil_attempts_per_operating_condition` 限制该层候选生成总数。达到成功配额后停止该层；达到尝试上限仍不足时立即报错，并保留已有记录用于检查或恢复。

每个条件锚点按照 `existing_to_sampled_ratios` 周期性选择经验条件或该层内 CL/CM 协方差条件。每个锚点最多生成 `noise_samples_per_condition` 个潜变量候选。超出成功配额的收敛候选保留为 `quota_excess_success` 审计记录，但不得进入累计训练集、条件统计或判别器真实样本池。失败和预处理失败同样消耗该层一次尝试。

若一个工况层在当前锚点的候选执行后仍未达到成功配额，下一 collection 周期会递增该层的 `anchor_index`：按比例重新选择条件来源、重新采样经验或协方差条件，并为新条件生成一组新的潜变量噪声。一个锚点内的多个候选共享条件但使用不同噪声。条件和噪声都由轮次、工况层、锚点及噪声索引组成的确定性种子生成，因此正常推进会得到新候选，断点恢复则会精确复现尚未持久化的请求。

每轮报告记录成功数、尝试数、每层状态及实际 XFoil 条件。历史已完成轮次不追溯重平衡；新的未完成轮次使用本定义。

采样状态在恢复时、每轮采样结束时和每轮训练结束时强制写入。采样进行中，每完成 `checkpoint_interval_collections` 个 collection 周期才写入一次累积数据集；一个 collection 周期是对所有尚未达到配额的 `(alpha, Re)` 分层各生成一批候选并完成 XFoil 的过程。收集器内部不得写入，因为此时 `accepted`/`quota_excess_success` 配额状态尚未确定。异常退出时，最多重做一个检查点间隔内的 XFoil 请求。

## 重置交替微调

`reset_finetuning.py` 的产物清单唯一来源于 `alternating_finetuning.ALTERNATING_RESET_ARTIFACTS`。它只删除累计生成数据集 `model/alternating_generated_dataset.pt`、交替轮次 checkpoint 目录 `model/alternating_checkpoints/` 和报告目录 `reports/alternating/`。初始 GAN、初始代理模型和原始数据集不在清理范围内。

不带参数时只列出受影响的现有路径。使用 `--confirm` 后才执行删除；下一次运行 `alternating_finetuning.py --config config.yaml` 会创建新的累计状态并从第 0 轮开始。
