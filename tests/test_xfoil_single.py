import os
import sys
import numpy as np
import pytest
from unittest.mock import patch

# Add parent directory to path to import from foildata
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foildata.xfoil import run_xfoil, run_xfoil_single


SUCCESS_OUTPUT = '''
 a =  2.000      CL =  0.5121
Cm = -0.0611     CD =  0.00991   =>   CDf =  0.00608
'''


def test_batch_xfoil_repaneled_after_load():
    with patch(
        'foildata.xfoil._execute_xfoil',
        return_value=('', '', False),
    ) as execute:
        success, timed_out = run_xfoil('unit_test_airfoil.dat', 2e5, 0, 2, 1)

    commands = execute.call_args.args[0]
    assert not success
    assert not timed_out
    assert 'NORM\nLOAD unit_test_airfoil.dat\nPANE\n' in commands


def test_single_xfoil_requires_all_aerodynamic_coefficients():
    incomplete_output = ' a =  2.000      CL =  0.5121\n'
    with patch('foildata.xfoil._execute_xfoil', return_value=(incomplete_output, '', False)):
        result = run_xfoil_single(np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), 2e5, 2.0)
    assert result is None


def test_single_xfoil_rejects_terminal_convergence_failure():
    failed_output = SUCCESS_OUTPUT + '\n VISCAL:  Convergence failed\n'
    with patch('foildata.xfoil._execute_xfoil', return_value=(failed_output, '', False)):
        result = run_xfoil_single(np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), 2e5, 2.0, return_all=True)
    assert result is None


def test_single_xfoil_returns_complete_converged_result():
    with patch('foildata.xfoil._execute_xfoil', return_value=(SUCCESS_OUTPUT, '', False)):
        result = run_xfoil_single(np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), 2e5, 2.0, return_all=True)
    assert result == {'CL': 0.5121, 'CD': 0.00991, 'CM': -0.0611}


def test_single_xfoil_continuation_steps_from_zero_and_scales_timeout():
    continuation_output = '''
 a =  0.000      CL =  0.1000
Cm = -0.0100     CD =  0.00800
 a =  1.000      CL =  0.3000
Cm = -0.0300     CD =  0.00900
 a =  2.000      CL =  0.5121
Cm = -0.0611     CD =  0.00991
'''
    with patch(
        'foildata.xfoil._execute_xfoil',
        return_value=(continuation_output, '', False),
    ) as execute:
        result = run_xfoil_single(
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            2e5,
            2.0,
            timeout=2,
            return_all=True,
            alpha_continuation=True,
        )

    commands = execute.call_args.args[0]
    assert 'NORM\nLOAD temp_foil_' in commands
    assert '\nPANE\n' in commands
    assert 'ASEQ 0 2 1' in commands
    assert 'VACC 0' in commands
    assert execute.call_args.kwargs['timeout'] == 6
    assert result == {'CL': 0.5121, 'CD': 0.00991, 'CM': -0.0611}


def test_single_xfoil_continuation_ignores_intermediate_alpha_failure():
    continuation_output = '''
 a =  0.000      CL =  0.1000
Cm = -0.0100     CD =  0.00800
 VISCAL:  Convergence failed
 a =  1.000      CL =  0.3000
Cm = -0.0300     CD =  0.00900
'''
    with patch(
        'foildata.xfoil._execute_xfoil',
        return_value=(continuation_output, '', False),
    ):
        result = run_xfoil_single(
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            2e5,
            1.0,
            return_all=True,
            alpha_continuation=True,
        )

    assert result == {'CL': 0.3, 'CD': 0.009, 'CM': -0.03}


def test_single_xfoil_continuation_rejects_output_without_target_alpha():
    with patch('foildata.xfoil._execute_xfoil', return_value=(SUCCESS_OUTPUT, '', False)):
        result = run_xfoil_single(
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            2e5,
            3.0,
            return_all=True,
            alpha_continuation=True,
        )
    assert result is None


@pytest.mark.parametrize('alpha', [-1.0, 2.5])
def test_single_xfoil_continuation_rejects_unsupported_target_alpha(alpha):
    with pytest.raises(ValueError):
        run_xfoil_single(
            np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            2e5,
            alpha,
            alpha_continuation=True,
        )

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
