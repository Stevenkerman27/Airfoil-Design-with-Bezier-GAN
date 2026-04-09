import os
import sys
import numpy as np

# Add parent directory to path to import from foildata
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foildata.xfoil import run_xfoil_single

def test_naca2412():
    # Load coordinates
    # Using np.loadtxt with a try-except to handle potential header lines
    dat_path = os.path.join(os.path.dirname(__file__), "naca2412.dat")
    try:
        coords = np.loadtxt(dat_path, skiprows=1)
    except:
        # If the file has a different structure, we'll parse it manually
        with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        parsed_coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    parsed_coords.append([x, y])
                except ValueError:
                    pass
        coords = np.array(parsed_coords)
    
    print(f"Loaded {len(coords)} points for NACA 2412.")
    
    # Test conditions from tests/2412
    # Mach = 0.000, Re = 0.200 e 6 = 200,000, alpha = 2.000
    reynolds = 200000
    alpha = 2.0
    expected_cl = 0.5121
    
    print(f"Running run_xfoil_single at Re={reynolds}, alpha={alpha}")
    
    # run xfoil
    cl = run_xfoil_single(coords, reynolds, alpha)
    
    print(f"Expected Cl: {expected_cl}")
    print(f"Actual Cl:   {cl}")
    
    if cl is not None:
        diff = abs(cl - expected_cl)
        print(f"Absolute difference: {diff:.5f}")
        if diff < 0.01:
            print("SUCCESS: Result matches closely!")
        else:
            print("FAILURE: Result deviates significantly!")
    else:
        print("FAILURE: run_xfoil_single returned None")

if __name__ == "__main__":
    test_naca2412()