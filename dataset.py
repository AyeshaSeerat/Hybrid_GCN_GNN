import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split

# === Custom Dataset Class ===
class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# === Load Data from NPZ Files ===
def load_data_from_npz(data_dir):
    data_list = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            x = torch.tensor(data['x'], dtype=torch.float32)
            edge_index = torch.tensor(data['edgeindex'], dtype=torch.long)
            y = torch.tensor(data['y'], dtype=torch.long)
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(graph_data)
    print(f"Total graphs loaded: {len(data_list)}")
    return data_list

