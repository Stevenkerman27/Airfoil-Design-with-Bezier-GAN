import subprocess
import os
import yaml
import numpy as np
import random
import glob

# Define relative paths from foildata/
COORD_DIR = "processed_foil"
POLAR_DIR = "polars"
foil_n = 400

# Ensure output directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), POLAR_DIR), exist_ok=True)

def load_config():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_re_list(config):
    re_range = config.get('Re_range_step', [1e5, 8e5, 1e5])
    re_range = [float(x) for x in re_range]
    # [start, end, step] -> inclusive of end if possible
    return np.arange(re_range[0], re_range[1] + re_range[2]/2, re_range[2])

def _execute_xfoil(commands, cwd, timeout):
    """
    Helper function to execute xfoil commands via subprocess and handle timeouts.
    Returns: (stdout, stderr, is_timeout)
    """
    process = subprocess.Popen(
        ['xfoil'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd=cwd
    )
    
    try:
        stdout, stderr = process.communicate(input=commands, timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return stdout, stderr, True

def run_xfoil(airfoil_name, reynolds, alpha_start, alpha_end, alpha_step):
    """
    airfoil_name: .dat文件名
    """
    # Create a unique filename including Reynolds number
    # Remove .dat extension for the filename
    name_base = os.path.splitext(airfoil_name)[0]
    filename = f"{name_base}_Re{int(reynolds):d}_polar.txt"
    
    # Result path relative to COORD_DIR
    save_file_rel = f"../{POLAR_DIR}/{filename}"
    
    # Absolute path for checking/removing existing files
    base_dir = os.path.dirname(__file__)
    save_file_abs = os.path.join(base_dir, POLAR_DIR, filename)
    
    if os.path.exists(save_file_abs):
        os.remove(save_file_abs)

    commands = f"""
    NORM
    LOAD {airfoil_name}
    OPER
    ITER {20}
    VISC {reynolds}
    PACC
    {save_file_rel}
    
    ASEQ {alpha_start} {alpha_end} {alpha_step}
    
    QUIT
    """

    cwd = os.path.join(base_dir, COORD_DIR)
    stdout, _, is_timeout = _execute_xfoil(commands, cwd, timeout=30)
    
    if is_timeout:
        print(f"警告: {airfoil_name} 在 Re={reynolds} 下计算超时(30秒)，已中断并跳过")
        
    return stdout

def run_xfoil_single(coords, reynolds, alpha, timeout=5):
    """
    Evaluates a single airfoil using Xfoil.
    Returns the Cl value if successful, or None if it fails to converge.
    """
    import tempfile
    import uuid
    
    # Generate unique filename for the temporary coordinates
    temp_filename = f"temp_foil_{uuid.uuid4().hex[:8]}.dat"
    base_dir = os.path.dirname(__file__)
    coord_dir = os.path.join(base_dir, COORD_DIR)
    os.makedirs(coord_dir, exist_ok=True)
    temp_filepath = os.path.join(coord_dir, temp_filename)
    
    # Write coordinates to temp file
    try:
        with open(temp_filepath, 'w') as f:
            f.write(f"Temp Airfoil\n")
            for pt in coords:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
                
        commands = f"""
        NORM
        LOAD {temp_filename}
        OPER
        ITER 50
        VISC {reynolds}
        ALFA {alpha}
        QUIT
        """
        
        stdout, _, is_timeout = _execute_xfoil(commands, coord_dir, timeout=timeout)
        
        if is_timeout:
            return None
            
        # Parse output for Cl
        for line in reversed(stdout.split('\n')):
            if 'a =' in line and 'CL =' in line:
                parts = line.split()
                try:
                    cl_idx = parts.index('CL') + 2 # usually 'CL = 0.5000'
                    return float(parts[cl_idx])
                except (ValueError, IndexError):
                    pass
        return None
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

if __name__ == "__main__":
    config = load_config()
    
    # Alpha parameters
    alpha_cfg = config.get('alpha_range_step', [0, 8, 1])
    a_start, a_end, a_step = alpha_cfg
    
    # Reynolds numbers
    re_list = get_re_list(config)
    
    # Airfoil selection
    base_dir = os.path.dirname(__file__)
    coord_path = os.path.join(base_dir, COORD_DIR)
    all_foils = [os.path.basename(f) for f in glob.glob(os.path.join(coord_path, "*.dat"))]
    
    if len(all_foils) > foil_n:
        selected_foils = random.sample(all_foils, foil_n)
    else:
        selected_foils = all_foils

    print(f"Selected {len(selected_foils)} airfoils for analysis.")
    print(f"Reynolds numbers: {re_list}")

    for foil in selected_foils:
        for re in re_list:
            print(f"Processing Airfoil {foil} at Re={re:.1e}...")
            run_xfoil(foil, re, a_start, a_end, a_step)