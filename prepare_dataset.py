import glob
import os

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from surrogate_split import build_cross_validation_manifest
from artifact_io import save_report_figure


PROCESSED_AIRFOIL_DIRECTORY = os.path.join('foildata', 'processed_foil')
POLAR_DIRECTORY = os.path.join('foildata', 'polars')
SURROGATE_DATASET_PATH = 'model/airfoil_dataset.pt'
SURROGATE_SPLIT_PATH = 'model/surrogate_airfoil_group_cv_split.pt'

def plot_label_statistics(data_list, output_path="reports/dataset/label_statistics.png"):
    """
    绘制 CL, CM, CD 的箱线图，并叠加所有原始数据点。
    """
    if not data_list:
        print("警告: 数据集为空，跳过统计图绘制。")
        return
        
    y_labels = torch.stack([d["y"] for d in data_list]).numpy()
    cd_values = np.array([d["cd"] for d in data_list], dtype=np.float32).reshape(-1, 1)
    coeffs = np.concatenate([y_labels[:, 2:], cd_values], axis=1)
    label_names = ['CL', 'CM', 'CD']
    
    plt.figure(figsize=(24, 6))
    for i, label_name in enumerate(label_names):
        plt.subplot(1, len(label_names), i + 1)
        
        # 1. 绘制箱线图 (关闭离群点显示 showfliers=False，因为我们要手动画所有点)
        plt.boxplot(coeffs[:, i], widths=0.5, showfliers=False, 
                    patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.3))
        
        # 2. 生成水平抖动 (Jitter)
        # 箱线图默认在 x=1 的位置，我们让散点在 1 附近随机偏移
        x_jitter = np.random.normal(1, 0.04, size=len(coeffs[:, i]))
        
        # 3. 绘制所有散点
        # s 是点的大小，alpha 是透明度（数据多时建议调低），c 是颜色
        plt.scatter(x_jitter, coeffs[:, i], s=1, alpha=0.3, c='blue', label='Data Points')
        
        plt.title(label_name)
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_report_figure(plt.gcf(), output_path)
    print(f"包含所有数据点的统计图已保存至: {output_path}")

def prepare_dataset(processed_foil_dir, polars_dir, output_file, max_cd, config):
    """
    读取翼型坐标文件和极曲线数据，将坐标点展平为 1D 特征张量，
    并提取对应的条件标签 [alpha, Re, CL, CM]。
    """
    data_list = []
    
    # 获取所有的 polar.txt 文件
    polar_files = glob.glob(os.path.join(polars_dir, "*_polar.txt"))
    print("found "+str(len(polar_files))+" polar files!")
    
    for p_file in polar_files:
        basename = os.path.basename(p_file)
        # 解析文件名以获取翼型名称和雷诺数
        parts = basename.split('_Re')
        if len(parts) != 2:
            continue
        foil_name = parts[0]
        
        re_part = parts[1].replace('_polar.txt', '')
        try:
            Re = float(re_part)
        except ValueError:
            continue
            
        # 寻找对应的坐标文件
        foil_path = os.path.join(processed_foil_dir, f"{foil_name}.dat")
        if not os.path.exists(foil_path):
            # 处理部分带有下划线前缀的坐标文件
            foil_path = os.path.join(processed_foil_dir, f"_{foil_name}.dat")
            if not os.path.exists(foil_path):
                continue
                
        # 读取翼型坐标 (跳过第一行标题)
        try:
            coords = np.loadtxt(foil_path, skiprows=1)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to load coordinate file: {foil_path}") from exc
            
        # 读取 polar 数据
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(p_file, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                
        # 寻找数据开始的行（在 '------' 之后）
        start_idx = -1
        for i, line in enumerate(lines):
            if '------' in line:
                start_idx = i + 1
                break
                
        if start_idx == -1:
            continue
            
        # 解析每一行的迎角和气动参数
        for line in lines[start_idx:]:
            if not line.strip():
                continue
            vals = line.split()
            # 通常至少包含 alpha, CL, CD, CDp, CM 等
            if len(vals) < 5:
                continue
                
            try:
                alpha = float(vals[0])
                CL = float(vals[1])
                CD = float(vals[2])
                CM = float(vals[4])
            except ValueError:
                continue

            # 过滤 CD 过大的数据
            if max_cd is not None and CD > max_cd:
                continue
                
            # 展平拼接为 1D 张量 (仅坐标), 采用 x,y,x,y顺序
            coords_flat = coords.flatten()
            x_tensor = torch.tensor(coords_flat, dtype=torch.float32)
            
            # 标签：alpha, Re, CL, CM (1D 张量)
            y_tensor = torch.tensor([alpha, Re, CL, CM], dtype=torch.float32)
            
            # 保存到数据集列表中
            data_list.append({
                "x": x_tensor,            # [x1, y1, x2, y2, ...]
                "y": y_tensor,            # [alpha, Re, CL, CM]
                "cd": CD,
                "foil_id": foil_name,
            })
            
    print(f"\n数据集准备完成！总共收集了 {len(data_list)} 个样本。")
    
    # 绘制标签统计图
    plot_label_statistics(data_list)
    
    print(f"正在保存至 {output_file} ...")
    torch.save(data_list, output_file)
    if SURROGATE_DATASET_PATH != output_file:
        raise ValueError(
            f'output_file must equal {SURROGATE_DATASET_PATH}, got {output_file}'
        )
    manifest = build_cross_validation_manifest(
        data_list,
        config['surrogate_test_ratio'],
        config['surrogate_cv_fold_count'],
        config['surrogate_seed'],
    )
    split_path = SURROGATE_SPLIT_PATH
    torch.save(manifest, split_path)
    print(f"已保存代理模型五折划分清单至 {split_path}")
    fold_sizes = [len(indices) for indices in manifest['fold_indices']]
    print(
        f"开发集样本数: {len(manifest['development_indices'])}, "
        f"测试集样本数: {len(manifest['test_indices'])}, fold 样本数: {fold_sizes}"
    )
    print("保存成功！可以使用 torch.load('{}') 进行读取。".format(output_file))

if __name__ == '__main__':
    # 读取配置
    config_path = "config.yaml"
    max_cd = None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        max_cd = config['max_Cd'] # CD threshold can be optional
        if max_cd is not None:
            print(f"将过滤 CD > {max_cd} 的数据")

    # 按照当前目录结构设置路径
    processed_dir = PROCESSED_AIRFOIL_DIRECTORY
    polars_dir = POLAR_DIRECTORY
    out_file = SURROGATE_DATASET_PATH

    prepare_dataset(processed_dir, polars_dir, out_file, max_cd=max_cd, config=config)
