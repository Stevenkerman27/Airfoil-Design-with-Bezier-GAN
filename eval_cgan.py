import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Switch to non-interactive backend
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.stats import qmc
from model import Generator
from utils import calculate_relative_thickness
from foildata.xfoil import run_xfoil_single

# 默认配置
DEFAULT_N_COND = 60
DEFAULT_K_SAMPLES = 20
DEFAULT_TOP_M = 5

def get_dataset_stats(data_path):
    """从数据集中提取升力系数与迎角的线性关系及厚度范围"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found. Please run prepare_dataset.py first.")
    
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    y = np.array([item['y'].numpy() for item in data])
    
    alpha = y[:, 0]
    cl = y[:, 2]
    thick = y[:, 3]
    
    # 线性回归: CL = m * alpha + c
    m, c = np.polyfit(alpha, cl, 1)
    
    # 计算残差标准差作为生成时的噪声强度
    cl_pred = m * alpha + c
    std_res = np.std(cl - cl_pred)
    
    return {
        'm': m,
        'c': c,
        'std_res': std_res,
        'thick_min': thick.min(),
        'thick_max': thick.max()
    }

def plot_cl_alpha_relationship(ds_stats, cond_samples, alpha_range, cl_noise_std, filename):
    """绘制迎角与升力系数的关系图，包含线性拟合线和生成的采样点范围"""
    plt.figure(figsize=(10, 6))
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], 100)
    cl_line = ds_stats['m'] * alphas + ds_stats['c']
    
    # 1. 绘制线性关系直线
    plt.plot(alphas, cl_line, 'r-', linewidth=2, label=f'Linear Fit: CL = {ds_stats["m"]:.4f}*α + {ds_stats["c"]:.4f}')
    
    # 2. 绘制生成的升力系数范围 (半透明区域，表示采样时的噪声分布)
    # 使用 2 * cl_noise_std 作为范围，与生成逻辑保持一致
    plt.fill_between(alphas, cl_line - 2*cl_noise_std, cl_line + 2*cl_noise_std, 
                     color='blue', alpha=0.1, label=f'Target CL Generation Range (±2σ, σ={cl_noise_std:.3f})')
    
    # 3. 绘制实际生成的评估采样点
    plt.scatter(cond_samples[:, 0], cond_samples[:, 2], color='blue', s=20, alpha=0.4, label='Generated Evaluation Targets')
    
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Lift Coefficient (CL)')
    plt.title('Evaluation Condition Distribution: CL vs Alpha')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_heatmap(x, y, z, title, filename, cmap='jet'):
    plt.figure(figsize=(9, 7))
    # 使用 tricontourf 生成平滑的等高线图
    # levels 增加平滑度
    levels = 20
    try:
        cntr = plt.tricontourf(x, y, z, levels=levels, cmap=cmap)
        plt.colorbar(cntr, label='Inaccuracy (Mean Error % + Variance)')
    except Exception as e:
        print(f"Warning: Could not create contour plot for {title}: {e}. Falling back to scatter.")
        sc = plt.scatter(x, y, c=z, cmap=cmap, edgecolors='k', alpha=0.8)
        plt.colorbar(sc, label='Inaccuracy (Mean Error % + Variance)')
    
    # 叠加散点显示原始采样点位置
    plt.scatter(x, y, c='k', s=10, alpha=0.4)
    
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Reynolds Number')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def _worker_eval_xfoil(args):
    """Worker function for parallel XFoil evaluation"""
    coords, re_input, alpha_input, target_cl, target_thick = args
    
    thickness = calculate_relative_thickness(coords)
    # 使用绝对误差
    thick_err = abs(thickness - target_thick)
    
    xfoil_res = run_xfoil_single(coords, re_input, alpha_input, return_all=True)
    
    cl = xfoil_res.get('CL', np.nan) if xfoil_res else np.nan
    cd = xfoil_res.get('CD', np.nan) if xfoil_res else np.nan
    cm = xfoil_res.get('CM', np.nan) if xfoil_res else np.nan

    # 只有当 CL, CD, CM 全部有效时才计算误差
    success = not np.isnan(cl) and not np.isnan(cd) and not np.isnan(cm)
    if success:
        cl_err = abs(cl - target_cl)
    else:
        # 失败时的绝对误差惩罚
        cl_err = 0.5 
        thick_err = max(thick_err, 0.05)
        
    total_err = thick_err + cl_err #用于Top M排序
    
    return {
        'coords': coords, 'total_err': total_err, 'thick_err': thick_err, 'cl_err': cl_err,
        'thickness': thickness, 'cl': cl, 'cd': cd, 'cm': cm,
        'alpha': alpha_input, 're': re_input, 'target_thick': target_thick, 'target_cl': target_cl,
        'success': success
    }

def evaluate_model(model_path, tag, config, device, cond_mean, cond_std, coord_norm_stats, n_cond, k_samples, top_m, ds_stats):
    print(f"\n--- Evaluating {tag} model: {model_path} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found.")

    generator = Generator(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()

    # Load coordinate normalization stats
    x_min = coord_norm_stats['x_min'].to(device)
    x_max = coord_norm_stats['x_max'].to(device)
    y_min = coord_norm_stats['y_min'].to(device)
    y_max = coord_norm_stats['y_max'].to(device)

    # Generate LHS samples for Alpha, Re, Thickness (3D)
    alpha_range = config.get('alpha_range_step')
    re_range = config.get('Re_range_step')
    
    # [Alpha, Re, Thickness]
    bounds_3d = np.array([
        [float(alpha_range[0]), float(alpha_range[1])],
        [float(re_range[0]), float(re_range[1])],
        [float(ds_stats['thick_min']), float(ds_stats['thick_max'])]
    ], dtype=float)

    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_cond)
    scaled_samples = qmc.scale(sample, bounds_3d[:, 0], bounds_3d[:, 1])

    # Calculate CL and construct 4D conditions: [Alpha, Re, CL, Thickness]
    cl_noise_std = config.get('cl_noise_std')
    cond_samples = []
    for s in scaled_samples:
        alpha_input, re_input, target_thick = s
        # target_cl = m * alpha + c + random_error
        target_cl = ds_stats['m'] * alpha_input + ds_stats['c'] + np.random.normal(0, cl_noise_std)#均值为 0，标准差为sigma的正态分布
        cond_samples.append([alpha_input, re_input, target_cl, target_thick])
    
    cond_samples = np.array(cond_samples)

    # 绘制迎角与升力系数的关系图
    os.makedirs('model', exist_ok=True)
    plot_cl_alpha_relationship(ds_stats, cond_samples, [alpha_range[0], alpha_range[1]], cl_noise_std, f'model/eval_{tag.lower()}_cl_alpha.png')

    noise_dim = config['noise_dimension']
    num_output_points = config['num_output_points']
    max_workers = config['max_workers']

    # Keep track of results for each condition
    results_by_cond = [[] for _ in range(n_cond)]
    max_retries = 5  # Maximum number of batches to attempt
    
    print(f"Starting iterative evaluation to collect {k_samples} successful samples for each of {n_cond} conditions...")
    
    for attempt in range(max_retries):
        active_indices = [i for i, res_list in enumerate(results_by_cond) if len(res_list) < k_samples]
        if not active_indices:
            break
            
        print(f"Attempt {attempt+1}/{max_retries}: {len(active_indices)} conditions need more samples.")
        
        # Prepare batch for generation
        batch_conds = []
        batch_indices = []
        for i in active_indices:
            needed = k_samples - len(results_by_cond[i])
            for _ in range(needed):
                batch_conds.append(cond_samples[i])
                batch_indices.append(i)
        
        batch_conds = np.array(batch_conds)
        num_to_gen = len(batch_conds)
        
        # Generator inference
        cond_tensor = torch.tensor(batch_conds, dtype=torch.float32).to(device)
        norm_cond = (cond_tensor - cond_mean) / cond_std
        noise = torch.randn(num_to_gen, noise_dim).to(device)
        
        with torch.no_grad():
            gen_out = generator(noise, norm_cond)
        
        # Un-normalize coordinates
        gen_out = gen_out.view(num_to_gen, num_output_points, 2)
        gen_out[:, :, 0] = gen_out[:, :, 0] * (x_max - x_min + 1e-8) + x_min
        gen_out[:, :, 1] = gen_out[:, :, 1] * (y_max - y_min + 1e-8) + y_min
        gen_airfoils = gen_out.cpu().numpy()
        
        # Prepare parallel XFoil tasks
        eval_tasks = []
        for k in range(num_to_gen):
            alpha_input, re_input, target_cl, target_thick = batch_conds[k]
            eval_tasks.append((gen_airfoils[k], re_input, alpha_input, target_cl, target_thick))
            
        # Run XFoil evaluations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(_worker_eval_xfoil, eval_tasks))
            
        # Assign results to conditions
        for k, res in enumerate(batch_results):
            idx = batch_indices[k]
            if res['success']:
                if len(results_by_cond[idx]) < k_samples:
                    results_by_cond[idx].append(res)
            elif attempt == max_retries - 1:
                # If last attempt and still no success, add the failure to maintain sample count
                if len(results_by_cond[idx]) < k_samples:
                    results_by_cond[idx].append(res)

    # Flatten results for some downstream tasks (like top M)
    all_results = []
    for res_list in results_by_cond:
        all_results.extend(res_list)

    # Process results back into heatmaps data
    results_alpha, results_re = [], []
    results_thick_inacc, results_cl_inacc = [], []
    var_weight = config.get('eval_var_weight')
    
    for i in range(n_cond):
        batch = results_by_cond[i]
        if not batch: continue 
        
        batch_thick_errs = [r['thick_err'] for r in batch]
        batch_cl_errs = [r['cl_err'] for r in batch]
        
        thick_inacc = np.mean(batch_thick_errs) + var_weight * np.var(batch_thick_errs)
        cl_inacc = np.mean(batch_cl_errs) + var_weight * np.var(batch_cl_errs)
        
        # Take metadata from first sample of batch
        meta = batch[0]
        results_alpha.append(meta['alpha'])
        results_re.append(meta['re'])
        results_thick_inacc.append(thick_inacc)
        results_cl_inacc.append(cl_inacc)

    # Plot heatmaps
    os.makedirs('model', exist_ok=True)
    plot_heatmap(results_alpha, results_re, results_thick_inacc, f'{tag} Thickness Inaccuracy', f'model/eval_{tag.lower()}_thick.png')
    plot_heatmap(results_alpha, results_re, results_cl_inacc, f'{tag} CL Inaccuracy', f'model/eval_{tag.lower()}_cl.png')
                 
    # Save top M
    os.makedirs('foildata/gen', exist_ok=True)
    all_results.sort(key=lambda x: x['total_err'])
    print(f"\nSaving Top {top_m} {tag} airfoils...")
    
    for i, item in enumerate(all_results[:top_m]):
        filename = f"{tag}_Top{i+1}_Terr{item['thick_err']:.1f}_Clerr{item['cl_err']:.1f}_T{item['thickness']:.4f}_Cl{item['cl']:.4f}_Cd{item['cd']:.5f}.dat"
        filepath = os.path.join('foildata/gen', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            header = f"{tag}_Top{i+1}_Terr_{item['thick_err']:.1f}_Clerr_{item['cl_err']:.1f}_Thick_{item['thickness']:.4f}_Cl_{item['cl']:.4f}_Cd_{item['cd']:.5f}"
            f.write(header + "\n")
            for pt in item['coords']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        print(f"Saved: {filename}")

    # Calculate and print global statistics
    valid_results = [r for r in all_results if r['success']]
    if valid_results:
        avg_thick_err = np.mean([r['thick_err'] for r in valid_results])
        avg_cl_err = np.mean([r['cl_err'] for r in valid_results])
        
        # 为了方便理解，计算全局平均相对误差 (仅供参考)
        mean_target_thick = np.mean([r['target_thick'] for r in valid_results])
        mean_target_cl = np.mean([np.abs(r['target_cl']) for r in valid_results])
        rel_thick_err = (avg_thick_err / (mean_target_thick + 1e-4)) * 100
        rel_cl_err = (avg_cl_err / (mean_target_cl + 1e-4)) * 100

        print(f"\n--- Global Performance Statistics ({tag}) ---")
        print(f"Total Valid Samples: {len(valid_results)}/{len(all_results)}")
        print(f"Mean Thickness Abs Error: {avg_thick_err:.4f}  (Rel: {rel_thick_err:.2f}%)")
        print(f"Mean CL Abs Error:        {avg_cl_err:.3f}   (Rel: {rel_cl_err:.2f}%)")
        print(f"------------------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cond", type=int, default=DEFAULT_N_COND, help="Number of LHS conditions")
    parser.add_argument("--k_samples", type=int, default=DEFAULT_K_SAMPLES, help="Airfoils per condition")
    parser.add_argument("--top_m", type=int, default=DEFAULT_TOP_M, help="Top M airfoils to save")
    args = parser.parse_args()

    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device_cfg = config.get("device")
    if device_cfg.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_params = torch.load('model/cond_norm.pt', map_location=device, weights_only=True)
    coord_norm_stats = torch.load('model/coord_norm.pt', map_location=device, weights_only=True)
    cond_mean = norm_params['mean'].to(device)
    cond_std = norm_params['std'].to(device)

    # 从数据集中提取统计信息
    ds_stats = get_dataset_stats('model/airfoil_dataset.pt')

    evaluate_model('model/pre_train.pt', 'PRE', config, device, cond_mean, cond_std, coord_norm_stats, args.n_cond, args.k_samples, args.top_m, ds_stats)
    evaluate_model('model/gan_final.pt', 'PG', config, device, cond_mean, cond_std, coord_norm_stats, args.n_cond, args.k_samples, args.top_m, ds_stats)
