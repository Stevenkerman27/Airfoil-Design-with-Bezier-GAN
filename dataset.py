import torch
from torch.utils.data import Dataset

class AirfoilDataset(Dataset):
    def __init__(self, data_path, norm_path="model/cond_norm.pt"):
        raw_data = torch.load(data_path)
        
        y_all = torch.stack([item['y'] for item in raw_data])
        self.y_mean = y_all.mean(dim=0)
        self.y_std = y_all.std(dim=0) + 1e-8
        
        # Save normalization parameters
        torch.save({'mean': self.y_mean, 'std': self.y_std}, norm_path)
        
        self.data = []
        for item in raw_data:
            norm_y = (item['y'] - self.y_mean) / self.y_std
            self.data.append({
                'x': item['x'],
                'y': norm_y
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']