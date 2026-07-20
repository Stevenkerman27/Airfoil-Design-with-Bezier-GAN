import os
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import yaml

# Add root directory to sys.path to import model and utils
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from cst import split_surface_t_values
from utils import calculate_relative_thickness, normalize_airfoil_chord_coordinates


FILES_TO_DELETE = [
    '30p-30n.dat',
    '30p-30n-flap.dat',
    '30p-30n-main.dat',
    '30p-30n-slat.dat',
    'e376.dat',
    'e377.dat',
    'e377m.dat',
    'e378.dat',
    'e379.dat',
    's1210.dat',
    's1211.dat',
    's1221-4deg-flap.dat',
    's1223.dat',
    's1223rtl.dat',
    'as6091.dat',
    'as6092.dat',
    'as6093.dat',
    'as6094.dat',
    'as6095.dat',
    'as6096.dat',
    'as6097.dat',
    'as6098.dat',
    'as6099.dat',
    'goe531.dat',
    'ua2-180.dat',
    's1221.dat',
    'goe451.dat',
    'dsma523a.dat',
    'naca1.dat',
    'ua79sff.dat',
    'ua79sfm.dat',
    'r1145msf.dat',
    'r1145msm.dat',
    'goe802a.dat',
    'goe802b.dat',
    'goe388.dat',
    'eiffel10.dat',
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
        
    num_points = config['num_output_points']
    beta = config['point_density_beta']
    preprocess_config = config['airfoil_preprocess']
    max_relative_thickness = preprocess_config['max_relative_thickness']
            
    target_dir.mkdir(parents=True, exist_ok=True)
    dat_files = list(source_dir.glob("*.dat"))
    
    success_count = 0
    fail_count = 0
    thick_count = 0
    thick_files = []
    generated_names = set()
    
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
            
        status, rel_thickness = resample_single_airfoil(
            file_path,
            target_dir,
            num_points,
            beta,
            max_relative_thickness,
            output_name,
        )
        if status is True:
            success_count += 1
            generated_names.add(output_name)
        elif status == "thick":
            thick_count += 1
            thick_files.append((filename, rel_thickness))
        else:
            fail_count += 1

    stale_paths = [
        path for path in target_dir.glob('*.dat')
        if path.name not in generated_names
    ]
    for path in stale_paths:
        path.unlink()
            
    if thick_files:
        print("\nSkipped files due to high relative thickness:")
        for name, t in thick_files:
            print(f"  {name}: {t:.2%}")
            
    print(
        f"Processing complete. Success: {success_count}, Failed: {fail_count}, "
        f"Skipped (thick): {thick_count}, Removed stale: {len(stale_paths)}"
    )

def resample_single_airfoil(
    file_path,
    target_dir,
    num_points,
    beta,
    max_relative_thickness,
    output_name=None,
):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 2:
            return False, 0
            
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
            return False, 0
            
        coords = np.array(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        # Calculate cumulative arc length
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)
        
        if s[-1] == 0:
            return False, 0
            
        s = s / s[-1] # Normalize to [0, 1]

        # Remove duplicate s values to avoid interpolation error
        s_unique, idx = np.unique(s, return_index=True)
        x_unique = x[idx]
        y_unique = y[idx]
        
        if len(s_unique) < 2:
            return False, 0

        interp_x = interp1d(s_unique, x_unique, kind='linear')
        interp_y = interp1d(s_unique, y_unique, kind='linear')
        
        # Find s_le at minimum x
        idx_le = np.argmin(x)
        s_le = float(s[idx_le])
        
        if not 0.0 < s_le < 1.0:
            raise ValueError(
                f'Leading-edge arc-length position must be in (0, 1), got {s_le}'
            )
        upper_t, lower_t = split_surface_t_values(num_points, beta)
        upper_s = s_le * upper_t.numpy()
        lower_s = s_le + (1.0 - s_le) * lower_t.numpy()
        t_new = np.concatenate([upper_s, lower_s[1:]])
        
        resampled_x = interp_x(t_new)
        resampled_y = interp_y(t_new)
        resampled_coords = np.column_stack([resampled_x, resampled_y])
        expected_leading_edge_index = upper_t.shape[0] - 1
        leading_edge_index = int(np.argmin(resampled_coords[:, 0]))
        if leading_edge_index != expected_leading_edge_index:
            raise ValueError(
                'Resampled leading edge must be the shared split midpoint: '
                f'expected {expected_leading_edge_index}, got {leading_edge_index}'
            )
        resampled_coords = normalize_airfoil_chord_coordinates(
            resampled_coords,
            leading_edge_index,
        )
        rel_thickness = calculate_relative_thickness(resampled_coords)
        if rel_thickness > max_relative_thickness:
            return "thick", rel_thickness
        
        # Ensure target dir exists
        out_name = output_name if output_name else file_path.name
        out_path = target_dir / out_name
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for rx, ry in resampled_coords:
                f.write(f" {rx:10.6f} {ry:10.6f}\n")
                
        return True, rel_thickness
    except Exception as exc:
        raise RuntimeError(f'Failed to resample {file_path}') from exc

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
