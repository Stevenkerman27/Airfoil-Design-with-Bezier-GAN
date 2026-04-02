import torch
import torch.optim as optim
import yaml
import os
import matplotlib.pyplot as plt
from model import BezierDecoderLayer

def load_dat(file_path):
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])
                except ValueError:
                    pass
    return torch.tensor(points, dtype=torch.float32)

def visualize_result(target_points, curve_points, control_points, weights, config, save_path='model/bezier_fit_result.png'):
    """
    可视化编码结果，并标注控制点权重
    """
    target_np = target_points.squeeze(0).cpu().numpy()
    curve_np = curve_points.squeeze(0).detach().cpu().numpy()
    cp_np = control_points.squeeze(0).detach().cpu().numpy()
    weights_np = weights.squeeze(0).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(target_np[:, 0], target_np[:, 1], 'k.', label='Original .dat', markersize=3)
    plt.plot(curve_np[:, 0], curve_np[:, 1], 'r-', label='Bezier Curve', linewidth=2)
    plt.plot(cp_np[:, 0], cp_np[:, 1], 'gx--', label='Control Points', alpha=0.6, markersize=5)
    
    # 标注每个控制点的权重
    for i, (x, y) in enumerate(cp_np):
        plt.text(x, y + 0.01, f'w={weights_np[i]:.2f}', fontsize=8, color='green', 
                 ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.axis('equal')
    plt.legend()
    plt.title(f"Airfoil Bezier Encoding ({config['num_control_points']} CP, {config['num_output_points']} Pts)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {save_path}")

def main():
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found.")
        return

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    target_path = "foildata/processed_foil/ag03.dat"
    if not os.path.exists(target_path):
        print(f"Error: {target_path} not found.")
        return
        
    target_points = load_dat(target_path).unsqueeze(0) # (1, K, 2)
    K = target_points.shape[1]
    
    N = config['num_control_points']
    # Initialize control points evenly along the target shape
    indices = torch.linspace(0, K - 1, N).long()
    init_cp = target_points[0, indices].clone()
    
    # Fix start and end points
    fixed_start_cp = target_points[:, 0:1, :] # (1, 1, 2)
    fixed_end_cp = target_points[:, -1:, :] # (1, 1, 2)
    
    # Intermediate control points are trainable
    inner_init_cp = init_cp[1:-1, :].unsqueeze(0) # (1, N-2, 2)
    trainable_control_points = torch.nn.Parameter(inner_init_cp)
    
    # All weights are trainable
    weights = torch.nn.Parameter(torch.ones((1, N), dtype=torch.float32))
    
    decoder = BezierDecoderLayer('config.yaml')
    
    lr = 0.01
    optimizer = optim.Adam([trainable_control_points, weights], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    iterations = 1200

    print(f"Starting optimization for {target_path}...")
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Concatenate fixed and trainable control points
        full_control_points = torch.cat([fixed_start_cp, trainable_control_points, fixed_end_cp], dim=1)
        abs_weights = torch.abs(weights)
        
        curve = decoder(full_control_points, abs_weights)
        if curve.shape[1] != target_points.shape[1]:
            raise IndexError(f"Generated points ({curve.shape[1]}) != target points ({target_points.shape[1]})")
            
        loss = torch.mean((curve - target_points) ** 2) * 20
        reg_loss = torch.mean(abs_weights**2) * 0.001
        
        # Penalty to keep control points relatively close
        cp_diff = full_control_points[:, 1:, :] - full_control_points[:, :-1, :]
        length_penalty = torch.mean(cp_diff ** 2) * 0.01
        
        total_loss = loss + reg_loss + length_penalty
        total_loss.backward()
        optimizer.step()
        
        scheduler.step(total_loss.item())
        
        if i % 100 == 0 or i == iterations - 1:
            print(f"Iter {i}, Total: {total_loss.item():.5f}, MSE: {loss.item():.5f}, Reg: {reg_loss.item():.5f}, Len:  {length_penalty.item():.5f}")
            
    # Save the result
    out_file = "model/encoded_bezier.pt"
    with torch.no_grad():
        final_cp = torch.cat([fixed_start_cp, trainable_control_points, fixed_end_cp], dim=1)
        final_w = torch.abs(weights)
        final_curve = decoder(final_cp, final_w)
        
    torch.save({
        'control_points': final_cp.detach().cpu(),
        'weights': final_w.detach().cpu()
    }, out_file)
    print(f"Encoded parameters saved to {out_file}")

    # 集成的可视化调用
    visualize_result(target_points, final_curve, final_cp, final_w, config)

if __name__ == '__main__':
    main()
