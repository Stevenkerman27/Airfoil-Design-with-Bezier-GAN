# 气动代理数据划分

`model/airfoil_dataset.pt` 是代理模型和 GAN 共用的原始数据集。项目只采用 `airfoil_group` 划分：`foil_id` 是不可分割的单位，同一翼型的全部工况样本不得跨开发集、五折验证集或独立测试集。

`prepare_dataset.py` 生成 `surrogate_dataset.split_path` 指向的版本化清单，其中包括：

- 独立测试集索引，比例为 `surrogate_test_ratio`。
- 开发集索引，即测试集之外的全部样本。
- `surrogate_cv_fold_count` 个互斥 fold，其并集恰好为开发集。

Optuna 仅在开发集内五折训练和选择超参数。最终代理模型与 GAN 训练均使用完整开发集；代理模型归一化参数也只由开发集计算。独立测试集仅供 `eval_surrogate.py` 和 GAN 的 XFoil 条件抽样使用。

修改数据源、`surrogate_test_ratio`、`surrogate_cv_fold_count` 或 `surrogate_seed` 后，必须重新生成数据集与划分清单：

```powershell
& 'C:\Users\zyx20\anaconda3\envs\myml\python.exe' prepare_dataset.py
```
