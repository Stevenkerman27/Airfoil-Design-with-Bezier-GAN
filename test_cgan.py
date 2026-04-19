import os
import torch
import yaml
import argparse
from model import Generator, Discriminator
import numpy as np
from utils import calculate_relative_thickness
from foildata.xfoil import run_xfoil_single

def generate_and_evaluate(model_path, tag, user_label_list):        
    print(f"\n--- Generating for {tag} using {model_path} ---")
    print(f"User defined label: {user_label_list}")
    
    alpha_input = user_label_list[0]
    re_input = user_label_list[1]
    
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
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} does not exist. Skipping.")
        return

    print(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
        # Assume it's a generator state dict if not a combined checkpoint
        generator.load_state_dict(checkpoint)
        print("Warning: Only generator weights found or loaded.")
    
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
    
    print(f"Generating airfoils for {tag}...")
    # 生成翼型和打分
    with torch.no_grad():
        generated_airfoils = generator(noise, cond) # (5, M*2)
        scores = discriminator(generated_airfoils, cond) # (5, 1)
        
    num_output_points = config.get('num_output_points', 60)
    
    generated_airfoils = generated_airfoils.view(5, num_output_points, 2).cpu().numpy()
    scores = scores.view(-1).cpu().numpy()
    
    print(f"{'No.':<4} | {'Score':<8} | {'Thick':<7} | {'CL':<8} | {'CD':<8} | {'CM':<8} | {'Status'}")
    print("-" * 70)
    
    for i in range(5):
        score = scores[i]
        airfoil_coords = generated_airfoils[i]
        
        # 计算生成翼型的实际厚度
        thickness = calculate_relative_thickness(airfoil_coords)
        
        # 按照判别器打分和厚度命名文件，并添加 PRE/PG 标识
        filename = f"{tag}_T{thickness:.4f}_S{score:.4f}.dat"
        filepath = os.path.join(save_dir, filename)
        
        # 将生成的坐标保存为 .dat 文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{tag}_Generated_Thickness_{thickness:.4f}_Score_{score:.4f}\n")
            for pt in airfoil_coords:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        
        # 调用 XFOIL 进行气动分析
        xfoil_res = run_xfoil_single(airfoil_coords, re_input, alpha_input, return_all=True)
        
        if xfoil_res:
            cl = xfoil_res.get('CL', np.nan)
            cd = xfoil_res.get('CD', np.nan)
            cm = xfoil_res.get('CM', np.nan)
            status = "Success"
        else:
            cl = cd = cm = np.nan
            status = "Failed"
            
        print(f"{i+1:<4} | {score:8.4f} | {thickness:7.4f} | {cl:8.4f} | {cd:8.5f} | {cm:8.4f} | {status}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate airfoils using Pre-train and PG models")
    parser.add_argument("--labels", "-l", type=float, nargs=4, help="Labels: Alpha Re Cl Thickness")
    args = parser.parse_args()
    
    custom_label = args.labels if args.labels else [2.0, 200000.0, 0.6, 0.12]
    
    # 分别为预训练模型和最终模型生成结果
    generate_and_evaluate('model/pre_train.pt', 'PRE', user_label_list=custom_label)
    generate_and_evaluate('model/gan_final.pt', 'PG', user_label_list=custom_label)
