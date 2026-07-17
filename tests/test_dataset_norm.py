import torch
import os
import numpy as np
import sys

# Add project root to path to import dataset
sys.path.append(os.getcwd())
from dataset import AirfoilDataset

def test_min_max_bounds():
    # Create dummy data
    dummy_data = [
        {'x': torch.tensor([0.0, -0.1, 1.0, 0.1]), 'y': torch.tensor([0, 0, 0, 0])}, # [x1, y1, x2, y2]
        {'x': torch.tensor([0.5, 0.2, 0.8, -0.05]), 'y': torch.tensor([0, 0, 0, 0])}
    ]
    os.makedirs("tests", exist_ok=True)
    torch.save(dummy_data, "tests/dummy_dataset.pt")
    
    ds = AirfoilDataset(
        "tests/dummy_dataset.pt",
        [0, 1],
        "tests/cond_norm.pt",
        "tests/coord_norm.pt",
    )
    x_norm, _ = ds[0]
    
    # Check bounds
    print(f"Normalized sample: {x_norm}")
    assert torch.all(x_norm >= -1e-7) # Floating point tolerance
    assert torch.all(x_norm <= 1.0000001)
    
    # Check stats file
    stats = torch.load("tests/coord_norm.pt")
    print(f"Stats: {stats}")
    assert 'x_min' in stats and 'y_max' in stats
    
    # Clean up
    os.remove("tests/dummy_dataset.pt")
    os.remove("tests/cond_norm.pt")
    os.remove("tests/coord_norm.pt")

if __name__ == "__main__":
    try:
        test_min_max_bounds()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
