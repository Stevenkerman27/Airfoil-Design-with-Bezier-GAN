import os
import torch
import yaml
import argparse
from model import Generator, Discriminator
import numpy as np
from train import (
    GAN_LABEL_ORDER,
    build_surrogate_conditions,
    load_frozen_surrogate,
    load_gan_auxiliary_stats,
    normalize_surrogate_coords,
)
from surrogate_split import load_cross_validation_manifest, resolve_surrogate_dataset_config
from utils import calculate_relative_thickness
from foildata.xfoil import run_xfoil_single

NUM_GENERATE = 5


def sample_development_conditions(raw_data, development_indices, count, seed):
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f'development sample count must be a positive integer, got {count}')
    if count > len(development_indices):
        raise ValueError(
            f'Requested {count} development conditions, only '
            f'{len(development_indices)} are available'
        )

    generator = torch.Generator().manual_seed(seed)
    selected_positions = torch.randperm(
        len(development_indices),
        generator=generator,
    )[:count].tolist()
    selected_conditions = []
    for position in selected_positions:
        dataset_index = development_indices[position]
        labels = raw_data[dataset_index]['y'].float()
        if labels.ndim != 1 or labels.numel() != len(GAN_LABEL_ORDER):
            raise ValueError(
                f'Dataset sample {dataset_index} must contain labels in order '
                f'{GAN_LABEL_ORDER}, got shape {tuple(labels.shape)}'
            )
        selected_conditions.append((dataset_index, labels.tolist()))
    return selected_conditions


def load_development_conditions(config, count):
    dataset_config = resolve_surrogate_dataset_config(config)
    raw_data = torch.load(dataset_config['data_path'], map_location='cpu', weights_only=True)
    manifest = load_cross_validation_manifest(raw_data, config)
    return sample_development_conditions(
        raw_data,
        manifest['development_indices'],
        count,
        config['surrogate_seed'],
    )


def generate_and_evaluate(model_path, tag, user_label_list, config):
    print(f"\n--- Generating for {tag} using {model_path} ---")
    print(f"User defined label: {user_label_list}")

    alpha_input = user_label_list[0]
    re_input = user_label_list[1]
    
    # 初始化设备
    device_cfg = config["device"]
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
    required_checkpoint_keys = {'generator_state_dict', 'discriminator_state_dict'}
    if not isinstance(checkpoint, dict) or not required_checkpoint_keys.issubset(checkpoint):
        raise ValueError(
            f'Model {model_path} is not a complete GAN checkpoint; expected keys '
            f'{sorted(required_checkpoint_keys)}'
        )
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    generator.eval()
    discriminator.eval()
    surrogate = load_frozen_surrogate(config, device)
    auxiliary_stats = load_gan_auxiliary_stats(config, device)
    
    # 加载条件归一化参数
    cond_mean = auxiliary_stats['gan_cond_mean']
    cond_std = auxiliary_stats['gan_cond_std']

    # 加载坐标归一化参数 (用于反归一化)
    gan_coord_stats = auxiliary_stats['gan_coord']
    x_min = gan_coord_stats['x_min']
    x_max = gan_coord_stats['x_max']
    y_min = gan_coord_stats['y_min']
    y_max = gan_coord_stats['y_max']
    
    # 处理用户输入的标签
    user_label = torch.tensor(user_label_list, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 归一化条件
    cond = (user_label - cond_mean) / cond_std
    # 扩展条件为 batch 大小
    cond = cond.expand(NUM_GENERATE, -1)
    
    # 随机生成噪声
    noise_dim = config['noise_dimension']
    # CGAN-GP 常见情况下，噪声可能是从标准正态分布采样
    noise = torch.randn(NUM_GENERATE, noise_dim).to(device)
    
    # 确保保存目录存在
    save_dir = 'foildata/gen'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating airfoils for {tag}...")
    # 生成翼型和打分
    with torch.no_grad():
        generated_airfoils = generator(noise, cond) # (NUM_GENERATE, M*2)
        scores = discriminator(generated_airfoils, cond) # (NUM_GENERATE, 1)
        
    num_output_points = config['num_output_points']
    
    # 转换为 (Batch, Points, 2) 格式
    generated_airfoils = generated_airfoils.view(NUM_GENERATE, num_output_points, 2)
    
    # 对坐标进行反归一化
    generated_airfoils[:, :, 0] = generated_airfoils[:, :, 0] * (x_max - x_min + 1e-8) + x_min
    generated_airfoils[:, :, 1] = generated_airfoils[:, :, 1] * (y_max - y_min + 1e-8) + y_min

    with torch.no_grad():
        surrogate_coords = normalize_surrogate_coords(
            generated_airfoils,
            auxiliary_stats['surrogate_coord'],
        )
        physical_conditions = user_label.expand(NUM_GENERATE, -1)
        surrogate_conditions = build_surrogate_conditions(
            physical_conditions,
            auxiliary_stats,
        )
        surrogate_targets = surrogate(surrogate_coords, surrogate_conditions)
        surrogate_targets = (
            surrogate_targets * auxiliary_stats['surrogate_target_std']
            + auxiliary_stats['surrogate_target_mean']
        )
    
    generated_airfoils = generated_airfoils.cpu().numpy()
    scores = scores.view(-1).cpu().numpy()
    surrogate_targets = surrogate_targets.cpu().numpy()
    
    target_cl = user_label_list[2]
    target_cm = user_label_list[3]
    if target_cl == 0 or target_cm == 0:
        raise ValueError(
            'Percentage error is undefined when the target CL or CM is zero.'
        )
    cm_weight, cl_weight = config['gan_surrogate_target_loss_weights']

    cl_errs = []
    cm_errs = []
    cl_pct_errs = []
    cm_pct_errs = []
    weighted_errs = []
    surrogate_cl_target_errs = []
    surrogate_cm_target_errs = []
    surrogate_cl_xfoil_errs = []
    surrogate_cm_xfoil_errs = []
    
    print(f"{'No.':<4} | {'Score':<8} | {'Thick':<7} | {'XF CL':<8} | {'Surr CL':<8} | {'XF CL Abs':<9} | {'XF CL %':<8} | {'S-X CL':<8} | {'XF CM':<8} | {'Surr CM':<8} | {'XF CM Abs':<9} | {'XF CM %':<8} | {'S-X CM':<8} | {'Weighted':<9} | {'CD':<8} | {'Status'}")
    print("-" * 204)
    
    for i in range(NUM_GENERATE):
        score = scores[i]
        airfoil_coords = generated_airfoils[i]
        surrogate_cm = surrogate_targets[i, 0]
        surrogate_cl = surrogate_targets[i, 1]
        surrogate_cl_target_err = abs(surrogate_cl - target_cl)
        surrogate_cm_target_err = abs(surrogate_cm - target_cm)
        surrogate_cl_target_errs.append(surrogate_cl_target_err)
        surrogate_cm_target_errs.append(surrogate_cm_target_err)
        
        # 计算生成翼型的实际厚度
        thickness = calculate_relative_thickness(airfoil_coords)
        
        # 调用 XFOIL 进行气动分析
        xfoil_res = run_xfoil_single(airfoil_coords, re_input, alpha_input, return_all=True)
        
        if xfoil_res:
            cl = xfoil_res.get('CL', np.nan)
            cl_err = abs(cl - target_cl)
            cl_pct_err = cl_err / abs(target_cl) * 100
            cl_errs.append(cl_err)
            cl_pct_errs.append(cl_pct_err)
            cd = xfoil_res.get('CD', np.nan)
            cm = xfoil_res.get('CM', np.nan)
            cm_err = abs(cm - target_cm)
            cm_pct_err = cm_err / abs(target_cm) * 100
            cm_errs.append(cm_err)
            cm_pct_errs.append(cm_pct_err)
            surrogate_cl_xfoil_err = abs(surrogate_cl - cl)
            surrogate_cm_xfoil_err = abs(surrogate_cm - cm)
            surrogate_cl_xfoil_errs.append(surrogate_cl_xfoil_err)
            surrogate_cm_xfoil_errs.append(surrogate_cm_xfoil_err)
            weighted_err = cm_weight * cm_err + cl_weight * cl_err
            weighted_errs.append(weighted_err)
            status = "Success"
        else:
            cl = cd = cm = np.nan
            cl_err = np.nan
            cl_pct_err = np.nan
            cm_err = np.nan
            cm_pct_err = np.nan
            surrogate_cl_xfoil_err = np.nan
            surrogate_cm_xfoil_err = np.nan
            weighted_err = np.nan
            status = "Failed"
            
        # 按照要求格式命名文件：type_Score_Thickness_Cl_Cd_Cm
        filename = f"{tag}_S{score:.4f}_T{thickness:.4f}_Cl{cl:.4f}_Cd{cd:.5f}_Cm{cm:.4f}.dat"
        filepath = os.path.join(save_dir, filename)
        
        # 将生成的坐标保存为 .dat 文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{tag}_Score_{score:.4f}_Thickness_{thickness:.4f}_Cl_{cl:.4f}_Cd_{cd:.5f}_Cm_{cm:.4f}\n")
            for pt in airfoil_coords:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        
        print(f"{i+1:<4} | {score:8.4f} | {thickness:7.4f} | {cl:8.4f} | {surrogate_cl:8.4f} | {cl_err:9.4f} | {cl_pct_err:7.2f}% | {surrogate_cl_xfoil_err:8.4f} | {cm:8.4f} | {surrogate_cm:8.4f} | {cm_err:9.4f} | {cm_pct_err:7.2f}% | {surrogate_cm_xfoil_err:8.4f} | {weighted_err:9.4f} | {cd:8.5f} | {status}")
        
    # 计算总体误差 (MAE)
    avg_cl_err = np.mean(cl_errs) if cl_errs else np.nan
    avg_cm_err = np.mean(cm_errs) if cm_errs else np.nan
    avg_cl_pct_err = np.mean(cl_pct_errs) if cl_pct_errs else np.nan
    avg_cm_pct_err = np.mean(cm_pct_errs) if cm_pct_errs else np.nan
    avg_weighted_err = np.mean(weighted_errs) if weighted_errs else np.nan
    avg_surrogate_cl_target_err = np.mean(surrogate_cl_target_errs)
    avg_surrogate_cm_target_err = np.mean(surrogate_cm_target_errs)
    avg_surrogate_cl_xfoil_err = np.mean(surrogate_cl_xfoil_errs) if surrogate_cl_xfoil_errs else np.nan
    avg_surrogate_cm_xfoil_err = np.mean(surrogate_cm_xfoil_errs) if surrogate_cm_xfoil_errs else np.nan

    print("-" * 204)
    print(
        f"Overall Batch MAE: CL={avg_cl_err:.5f} ({avg_cl_pct_err:.2f}%), "
        f"CM={avg_cm_err:.5f} ({avg_cm_pct_err:.2f}%), "
        f"Weighted={avg_weighted_err:.5f}"
    )
    print(
        f"Surrogate MAE to target: CL={avg_surrogate_cl_target_err:.5f}, "
        f"CM={avg_surrogate_cm_target_err:.5f}"
    )
    print(
        f"Surrogate MAE to XFoil: CL={avg_surrogate_cl_xfoil_err:.5f}, "
        f"CM={avg_surrogate_cm_xfoil_err:.5f}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate airfoils using a trained CWGAN-GP model")
    parser.add_argument("--model", "-m", type=str, default="model/gan_final.pt", help="Path to model checkpoint")
    parser.add_argument("--tag", type=str, default="GAN", help="Tag prefix used in generated filenames")
    parser.add_argument("--labels", "-l", type=float, nargs=4, help="Labels: Alpha Re Cl Cm")
    parser.add_argument(
        '--development-samples',
        type=int,
        default=5,
        help='Sample real development conditions without replacement; 0 uses --labels',
    )
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if args.development_samples < 0:
        raise ValueError(
            f'development sample count must be non-negative, got '
            f'{args.development_samples}'
        )
    if args.development_samples > 0:
        conditions = load_development_conditions(config, args.development_samples)
        print(
            f'Sampled {len(conditions)} development conditions with '
            f"seed {config['surrogate_seed']}"
        )
        for sample_number, (dataset_index, labels) in enumerate(conditions, start=1):
            condition_tag = f'{args.tag}_DEV{sample_number:03d}'
            print(f'Development dataset index: {dataset_index}')
            generate_and_evaluate(args.model, condition_tag, labels, config)
    else:
        custom_label = args.labels if args.labels else [2.0, 200000.0, 0.6, -0.2]
        generate_and_evaluate(args.model, args.tag, custom_label, config)
