import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_label_statistics(data_list, output_path="model/label_statistics.png"):
    """
    绘制 CL, CD, CM 的箱线图，并叠加所有原始数据点。
    """
    if not data_list:
        print("警告: 数据集为空，跳过统计图绘制。")
        return
        
    y_labels = torch.stack([d["y"] for d in data_list]).numpy()
    coeffs = y_labels[:, 2:] 
    label_names = ['CL', 'CD', 'CM']
    
    plt.figure(figsize=(15, 6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        
        # 1. 绘制箱线图 (关闭离群点显示 showfliers=False，因为我们要手动画所有点)
        plt.boxplot(coeffs[:, i], widths=0.5, showfliers=False, 
                    patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.3))
        
        # 2. 生成水平抖动 (Jitter)
        # 箱线图默认在 x=1 的位置，我们让散点在 1 附近随机偏移
        x_jitter = np.random.normal(1, 0.04, size=len(coeffs[:, i]))
        
        # 3. 绘制所有散点
        # s 是点的大小，alpha 是透明度（数据多时建议调低），c 是颜色
        plt.scatter(x_jitter, coeffs[:, i], s=1, alpha=0.3, c='blue', label='Data Points')
        
        plt.title(label_names[i])
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"包含所有数据点的统计图已保存至: {output_path}")

def prepare_dataset(processed_foil_dir, polars_dir, output_file="airfoil_dataset.pt"):
    """
    读取翼型坐标文件和极曲线数据，将坐标点展平为 1D 特征张量,并提取对应的气动参数（alpha, Re, CL, CD, CM）作为 1D 标签向量。
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
        except Exception:
            continue
            
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
                # vals[3] 是 CDp，跳过
                CM = float(vals[4])
            except ValueError:
                continue
                
            # 展平拼接为 1D 张量 (仅坐标), 采用 x,y,x,y顺序
            coords_flat = coords.flatten()
            x_tensor = torch.tensor(coords_flat, dtype=torch.float32)
            
            # 标签：alpha, Re, CL, CD, CM (1D 张量)
            y_tensor = torch.tensor([alpha, Re, CL, CD, CM], dtype=torch.float32)
            
            # 保存到数据集列表中
            data_list.append({
                "x": x_tensor,            # [x1, x2, ..., y1, y2, ...]
                "y": y_tensor             # [alpha, Re, CL, CD, CM]
            })
            
    print(f"\n数据集准备完成！总共收集了 {len(data_list)} 个样本。")
    
    # 绘制标签统计图
    plot_label_statistics(data_list)
    
    print(f"正在保存至 {output_file} ...")
    torch.save(data_list, output_file)
    print("保存成功！可以使用 torch.load('{}') 进行读取。".format(output_file))

if __name__ == '__main__':
    # 按照当前目录结构设置路径
    processed_dir = os.path.join("foildata", "processed_foil")
    polars_dir = os.path.join("foildata", "polars")
    out_file = "model/airfoil_dataset.pt"
    
    prepare_dataset(processed_dir, polars_dir, out_file)