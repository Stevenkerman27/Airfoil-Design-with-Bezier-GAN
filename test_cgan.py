import os
import torch
import yaml
from model import Generator, Discriminator
import numpy as np
from utils import calculate_relative_thickness

def generate_and_evaluate(user_label_list=None):
    if user_label_list is None:
        # 默认的用户自定义标签: [Alpha, Re, Cl, Thickness]
        user_label_list = [2.0, 200000.0, 0.6, 0.12]
        
    print(f"User defined label: {user_label_list}")
    
    # 读取配置
    config_path = "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 初始化设备
    device_cfg = config.get("device", "auto")
    if device_cfg.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 实例化模型
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    
    # 加载权重
    gen_weights = 'model/generator.pth'
    disc_weights = 'model/discriminator.pth'
    generator.load_state_dict(torch.load(gen_weights, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(disc_weights, map_location=device, weights_only=True))
    
    generator.eval()
    discriminator.eval()
    
    # 加载条件归一化参数
    norm_params = torch.load('model/cond_norm.pt', map_location=device, weights_only=True)
    cond_mean = norm_params['mean'].to(device)
    cond_std = norm_params['std'].to(device)
    
    # 处理用户输入的标签
    user_label = torch.tensor(user_label_list, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 归一化条件
    cond = (user_label - cond_mean) / cond_std
    # 扩展条件为5个batch，因为要生成5组不同的噪声
    cond = cond.expand(5, -1)
    
    # 随机生成5组噪声
    noise_dim = config.get('noise_dimension', 10)
    # CGAN-GP 常见情况下，噪声可能是从标准正态分布采样
    noise = torch.randn(5, noise_dim).to(device)
    
    # 确保保存目录存在
    save_dir = 'foildata/gen'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating airfoils...")
    # 生成翼型和打分
    with torch.no_grad():
        generated_airfoils = generator(noise, cond) # (5, M*2)
        scores = discriminator(generated_airfoils, cond) # (5, 1)
        
    num_output_points = config.get('num_output_points', 60)
    
    generated_airfoils = generated_airfoils.view(5, num_output_points, 2).cpu().numpy()
    scores = scores.view(-1).cpu().numpy()
    
    for i in range(5):
        score = scores[i]
        airfoil_coords = generated_airfoils[i]
        
        # 计算生成翼型的实际厚度
        thickness = calculate_relative_thickness(airfoil_coords)
        
        # 按照判别器打分和厚度命名文件
        filename = f"T{thickness:.4f}_S{score:.4f}.dat"
        filepath = os.path.join(save_dir, filename)
        
        # 将生成的坐标保存为 .dat 文件 (标准XFoil格式：两列，以空格/制表符分隔)
        # 第一行通常写翼型名字
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"CGAN_Generated_Thickness_{thickness:.4f}_Score_{score:.4f}\n")
            for pt in airfoil_coords:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        
        print(f"Saved generated airfoil {i+1}/5 to {filepath} (Thickness: {thickness:.4f}, Score: {score:.4f})")

if __name__ == '__main__':
    # 用户可以在此修改标签: [alpha, Re, Cl, Thickness]
    custom_label = [5.0, 400000.0, 0.8, 0.15]
    generate_and_evaluate(custom_label)
