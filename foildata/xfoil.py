import subprocess
import os
import yaml
import numpy as np
import random
import glob

# Define relative paths from foildata/
COORD_DIR = "processed_foil"
POLAR_DIR = "polars"
TEMP_DIR = "temp_foils"
foil_n = 1600

# Ensure output directories exist
os.makedirs(os.path.join(os.path.dirname(__file__), POLAR_DIR), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), TEMP_DIR), exist_ok=True)

def load_config():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_re_list(config):
    re_range = config['Re_range_step']
    re_range = [float(x) for x in re_range]
    # [start, end, step] -> inclusive of end if possible
    return np.arange(re_range[0], re_range[1] + re_range[2]/2, re_range[2])

def _execute_xfoil(commands, cwd, timeout):
    """
    Helper function to execute xfoil commands via subprocess and handle timeouts.
    Returns: (stdout, stderr, is_timeout)
    """
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        
    process = subprocess.Popen(
        ['xfoil'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd=cwd,
        startupinfo=startupinfo
    )
    
    try:
        stdout, stderr = process.communicate(input=commands, timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return stdout, stderr, True

def run_xfoil(airfoil_name, reynolds, alpha_start, alpha_end, alpha_step, timeout=10):
    """
    airfoil_name: .dat文件名
    Returns: (success, is_timeout)
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
    _, _, is_timeout = _execute_xfoil(commands, cwd, timeout=timeout)
    
    if is_timeout:
        print(f"警告: {airfoil_name} 在 Re={reynolds} 下计算超时({timeout}秒)，已中断并跳过")
        
    success = os.path.exists(save_file_abs) and os.path.getsize(save_file_abs) > 0
    return success, is_timeout

def run_xfoil_single(coords, reynolds, alpha, timeout=2, return_all=False):
    """
    Evaluates a single airfoil using Xfoil.
    Returns the Cl value if successful (or a dict of CL, CD, CM if return_all=True), 
    or None if it fails to converge.
    """
    import tempfile
    import uuid
    
    # Generate unique filename for the temporary coordinates
    temp_filename = f"temp_foil_{uuid.uuid4().hex[:8]}.dat"
    base_dir = os.path.dirname(__file__)
    temp_dir = os.path.join(base_dir, TEMP_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
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
        
        stdout, _, is_timeout = _execute_xfoil(commands, temp_dir, timeout=timeout)
        
        if is_timeout:
            return None
            
        # Parse output for Cl, Cd, Cm
        # XFOIL 6.99 output format:
        #        a =  2.000      CL =  0.5121
        #       Cm = -0.0611     CD =  0.00991   =>   CDf =  0.00608    CDp =  0.00383
        
        res = {}
        for line in reversed(stdout.split('\n')):
            line_upper = line.upper()
            parts = line.split()
            
            # Look for CD and Cm (usually on one line)
            if 'CD =' in line_upper or 'CM =' in line_upper:
                try:
                    # Handle both CD and CD = 
                    if 'CD' in parts:
                        res['CD'] = float(parts[parts.index('CD') + 2])
                    elif 'CD=' in parts:
                        res['CD'] = float(parts[parts.index('CD=') + 1])
                        
                    if 'Cm' in parts:
                        res['CM'] = float(parts[parts.index('Cm') + 2])
                    elif 'Cm=' in parts:
                        res['CM'] = float(parts[parts.index('Cm=') + 1])
                    elif 'CM' in parts:
                        res['CM'] = float(parts[parts.index('CM') + 2])
                except (ValueError, IndexError):
                    pass
            
            # Look for CL and alpha (usually on the preceding line in the output, 
            # so following line when reading reversed)
            if 'CL =' in line_upper:
                try:
                    if 'CL' in parts:
                        res['CL'] = float(parts[parts.index('CL') + 2])
                    elif 'CL=' in parts:
                        res['CL'] = float(parts[parts.index('CL=') + 1])
                    
                    if not return_all:
                        if 'CL' in res:
                            return res['CL']
                    else:
                        # If we found CL, we assume this is the most recent converged point
                        if 'CL' in res:
                            return res
                except (ValueError, IndexError):
                    pass
        return None
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

import concurrent.futures

def _worker_run_xfoil(args):
    foil, re, a_start, a_end, a_step, timeout = args
    success, is_timeout = run_xfoil(foil, re, a_start, a_end, a_step, timeout)
    return foil, re, success, is_timeout

if __name__ == "__main__":
    config = load_config()
    
    # Alpha parameters
    alpha_cfg = config['alpha_range_step']
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

    max_workers = config['max_workers']
    tasks = []
    for foil in selected_foils:
        for re in re_list:
            tasks.append((foil, re, a_start, a_end, a_step, 30))

    print(f"Starting parallel analysis with {max_workers} workers...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_run_xfoil, tasks))

    # Summary reporting
    timeouts = []
    failures = []
    for foil, re, success, is_timeout in results:
        if is_timeout:
            timeouts.append(f"{foil} (Re={re:.1e})")
        elif not success:
            failures.append(f"{foil} (Re={re:.1e})")

    print("\n--- Analysis Summary ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {len(tasks) - len(timeouts) - len(failures)}")
    
    if timeouts:
        print(f"\nTimeouts ({len(timeouts)}):")
        for t in timeouts:
            print(f"  - {t}")
            
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
