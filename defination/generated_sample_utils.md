# 生成样本公共工具

`generated_xfoil_utils.py` 是生成翼型 XFoil 记录的单一来源，负责把生成坐标变换到局部单位弦长坐标系、基于该坐标生成缓存键并执行严格收敛判定。记录中保存的规范化坐标直接交给 XFoil；代理模型也从同一份规范化几何构造输入，保证二者比较的是相同几何。所有 XFoil 调用均在 `NORM`、`LOAD` 后执行 `PANE`，只重分布 XFoil 内部求解面元，不改变记录坐标；生成翼型的单点调用还固定在 `VPAR` 中执行 `VACC 0`，以保留完整粘性耦合矩阵。成功条件必须由 XFoil 的实际 `CL/CM` 回填给调用方。`generated_surrogate_utils.py` 定义生成样本代理数据集及其统一评估指标。`artifact_io.py` 统一创建输出父目录并写入 YAML 报告。
