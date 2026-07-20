import unittest
import numpy as np
import sys
import os

# Add parent directory to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import calculate_relative_thickness

class TestUtils(unittest.TestCase):
    def test_naca0012_approx(self):
        # Rough NACA 0012-like coordinates
        x = np.linspace(0, 1, 11)
        # Simplified: max thickness 0.12 at x=0.3
        # Using a sine wave that gives roughly 0.12 thickness (0.06 amplitude for each surface)
        y_upper = 0.06 * np.sin(np.pi * x) 
        y_lower = -0.06 * np.sin(np.pi * x)
        
        # Combine into Selig format (1 -> 0 -> 1)
        coords = np.vstack([
            np.column_stack([x[::-1], y_upper[::-1]]),
            np.column_stack([x[1:], y_lower[1:]])
        ])
        
        thickness = calculate_relative_thickness(coords)
        # Expected: ~0.12
        self.assertAlmostEqual(thickness, 0.12, places=2)

    def test_thickness_compares_surfaces_at_the_same_x(self):
        coords = np.array([
            [1.0, 0.0],
            [0.25, 0.2],
            [0.0, 0.0],
            [0.75, -0.1],
            [1.0, 0.0],
        ])

        thickness = calculate_relative_thickness(coords)

        self.assertAlmostEqual(thickness, 7.0 / 30.0)

if __name__ == "__main__":
    unittest.main()
