import os
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import yaml

# Add root directory to sys.path to import model
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model import center_dense_spacing

# Configuration
FILES_TO_DELETE = [
    "30p-30n.dat",
    "30p-30n-flap.dat",
    "30p-30n-main.dat",
    "30p-30n-slat.dat",
    "e376.dat",
    "e377.dat",
    "e377m.dat",
    "e378.dat",
    "e379.dat",
    "s1210.dat",
    "s1211.dat",
    "s1221-4deg-flap.dat",
    "s1223.dat",
    "s1223rtl.dat",
    "as6091.dat",
    "as6092.dat",
    "as6093.dat",
    "as6094.dat",
    "as6095.dat",
    "as6096.dat",
    "as6097.dat",
    "as6098.dat",
    "as6099.dat",
]

def manage_files():
    # Set paths relative to script location
    base_dir = Path(__file__).parent
    target_dir = base_dir / "coord_seligFmt"
    processed_dir = base_dir / "processed_foil"
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} not found.")
        return

    print(f"Source directory: {target_dir.absolute()}")
    print(f"Target directory: {processed_dir.absolute()}")

    # 1. Processing Phase (Includes filtering + Renaming + Resampling)
    print("\n--- Processing Phase (Filtering + Renaming + Resampling) ---")
    resample_airfoils(target_dir, processed_dir)

    # 2. Validation Phase (Validate the final processed coordinates)
    validate_coordinates(processed_dir)

def resample_airfoils(source_dir, target_dir):
    # Load config to get target points
    config_path = root_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if 'num_output_points' not in config:
        raise KeyError("'num_output_points' must be specified in config.yaml")
    if 'point_density_beta' not in config:
        raise KeyError("'point_density_beta' must be specified in config.yaml")
        
    num_points = config['num_output_points']
    beta = config['point_density_beta']
            
    target_dir.mkdir(parents=True, exist_ok=True)
    dat_files = list(source_dir.glob("*.dat"))
    
    success_count = 0
    fail_count = 0
    
    for file_path in dat_files:
        filename = file_path.name
        
        # 1. Skip files in deletion list
        if filename in FILES_TO_DELETE:
            continue
            
        # 2. Determine output name based on naming logic
        if filename.lower().startswith(('t', 'f')) and not filename.startswith('_'):
            output_name = f"_{filename}"
        else:
            output_name = filename
            
        if resample_single_airfoil(file_path, target_dir, num_points, beta, output_name):
            success_count += 1
        else:
            fail_count += 1
            
    print(f"Processing complete. Success: {success_count}, Failed: {fail_count}")

def resample_single_airfoil(file_path, target_dir, num_points, beta=2.0, output_name=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 2:
            return False
            
        header = lines[0]
        coords = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        if not coords:
            return False
            
        coords = np.array(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Calculate cumulative arc length
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)
        
        if s[-1] == 0:
            return False
            
        s = s / s[-1] # Normalize to [0, 1]

        # Remove duplicate s values to avoid interpolation error
        s_unique, idx = np.unique(s, return_index=True)
        x_unique = x[idx]
        y_unique = y[idx]
        
        if len(s_unique) < 2:
            return False

        interp_x = interp1d(s_unique, x_unique, kind='linear')
        interp_y = interp1d(s_unique, y_unique, kind='linear')
        
        # Find s_le at minimum x
        idx_le = np.argmin(x)
        s_le = float(s[idx_le])
        
        # Get parameter t using the shared unified logic
        t_new_torch = center_dense_spacing(num_points, s_le=s_le, beta=beta)
        t_new = t_new_torch.numpy()
        
        resampled_x = interp_x(t_new)
        resampled_y = interp_y(t_new)
        
        # Ensure target dir exists
        out_name = output_name if output_name else file_path.name
        out_path = target_dir / out_name
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for rx, ry in zip(resampled_x, resampled_y):
                f.write(f" {rx:10.6f} {ry:10.6f}\n")
                
        return True
    except Exception as e:
        # print(f"Error resampling {file_path.name}: {e}")
        return False

def validate_coordinates(target_dir, tolerance=1e-2):
    print("\n--- Validation Phase ---")
    dat_files = list(target_dir.glob("*.dat"))
    if not dat_files:
        print("No .dat files found for validation.")
        return

    for file_path in dat_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if len(lines) < 2:
                print(f"[ERROR] {file_path.name}: File too short to contain coordinates.")
                continue

            # Skip header (first line)
            coords = lines[1:]
            
            def parse_line(line):
                # Handle potential non-numeric lines or malformed data
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[0]), float(parts[1])
                    except ValueError:
                        return None
                return None

            start_pt = parse_line(coords[0])
            end_pt = parse_line(coords[-1])

            for label, pt in [("Start", start_pt), ("End", end_pt)]:
                if pt:
                    x, y = pt
                    dx = abs(x - 1.0)
                    dy = abs(y - 0.0)
                    if dx > tolerance or dy > tolerance:
                        print(f"[WARNING] {file_path.name}: {label} point ({x:.6f}, {y:.6f}) - Discrepancy (dx={dx:.6f}, dy={dy:.6f})")
                elif label == "Start" or label == "End":
                    print(f"[ERROR] {file_path.name}: Could not parse {label.lower()} coordinate line.")

        except Exception as e:
            print(f"[ERROR] {file_path.name}: Failed to read/parse: {e}")

if __name__ == "__main__":
    manage_files()
