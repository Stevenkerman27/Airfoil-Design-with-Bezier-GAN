import subprocess
import os
import yaml
import numpy as np
import random
import glob

# Define relative paths from foildata/
COORD_DIR = "processed_foil"
POLAR_DIR = "polars"
foil_n = 100

# Ensure output directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), POLAR_DIR), exist_ok=True)

def load_config():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_dir, "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_re_list(config):
    re_range = config.get('Re_range_step', [1e5, 8e5, 1e5])
    re_range = [float(x) for x in re_range]
    # [start, end, step] -> inclusive of end if possible
    return np.arange(re_range[0], re_range[1] + re_range[2]/2, re_range[2])

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

    # Start process in the coordinate directory
    process = subprocess.Popen(
        ['xfoil'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd=os.path.join(base_dir, COORD_DIR)
    )
    
    stdout, stderr = process.communicate(input=commands)
    return stdout

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