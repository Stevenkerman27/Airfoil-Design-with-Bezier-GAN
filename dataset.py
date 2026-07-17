import torch
from torch.utils.data import Dataset

class AirfoilDataset(Dataset):
    def __init__(self, data_path, sample_indices, cond_norm_path, coord_norm_path):
        all_raw_data = torch.load(data_path, weights_only=True)
        if len(sample_indices) == 0:
            raise ValueError('GAN sample_indices must not be empty')
        raw_data = [all_raw_data[index] for index in sample_indices]
        
        # 1. Normalize conditions (labels) - Keep Z-score for conditions
        # Ensure floating point for mean/std calculation
        y_all = torch.stack([item['y'] for item in raw_data]).float()
        self.y_mean = y_all.mean(dim=0)
        self.y_std = y_all.std(dim=0) + 1e-8
        
        # Save condition normalization parameters
        torch.save({'mean': self.y_mean, 'std': self.y_std}, cond_norm_path)
        
        # 2. Normalize airfoil coordinates (Min-Max)
        coords_all = torch.stack([item['x'] for item in raw_data]).float() # (N, M*2)
        coords_all = coords_all.view(coords_all.size(0), -1, 2)
        
        self.x_min = coords_all[:, :, 0].min()
        self.x_max = coords_all[:, :, 0].max()
        self.y_min = coords_all[:, :, 1].min()
        self.y_max = coords_all[:, :, 1].max()
        
        # Save coordinate normalization parameters
        torch.save({
            'x_min': self.x_min, 'x_max': self.x_max,
            'y_min': self.y_min, 'y_max': self.y_max
        }, coord_norm_path)
        
        self.data = []
        for item in raw_data:
            # Normalize conditions
            norm_y = (item['y'].float() - self.y_mean) / self.y_std
            
            # Normalize x and y independently
            c = item['x'].float().clone().view(-1, 2)
            c[:, 0] = (c[:, 0] - self.x_min) / (self.x_max - self.x_min + 1e-8)
            c[:, 1] = (c[:, 1] - self.y_min) / (self.y_max - self.y_min + 1e-8)
            norm_x = c.view(-1)
            
            self.data.append({
                'x': norm_x,
                'y': norm_y
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']

if __name__ == "__main__":
    import os
    data_path = "model/airfoil_dataset.pt"
    if os.path.exists(data_path):
        print(f"Initializing dataset from {data_path}...")
        raw_data = torch.load(data_path, weights_only=True)
        ds = AirfoilDataset(
            data_path,
            list(range(len(raw_data))),
            'model/cond_norm.pt',
            'model/coord_norm.pt',
        )
        print("Dataset initialized and stats saved to model/cond_norm.pt and model/coord_norm.pt")
        
        # Verify stats
        coord_stats = torch.load("model/coord_norm.pt")
        print(f"Coord stats: {coord_stats}")
        
        x_sample, y_sample = ds[0]
        print(f"Sample x range: [{x_sample.min():.4f}, {x_sample.max():.4f}]")
    else:
        print(f"Error: {data_path} not found. Run prepare_dataset.py first.")
