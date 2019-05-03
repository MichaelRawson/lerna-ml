from functools import lru_cache
import json
import torch
from tqdm import tqdm
import os
from torch_geometric.data import Data, Dataset

def load_graph(path):
    with open(path, 'rb') as f:
        record = json.load(f)
        x = torch.tensor(record['nodes'], dtype=torch.float)
        edge_index = torch.tensor(record['edges'], dtype=torch.long).t()
        y = torch.tensor([int(record['y'])])
        return Data(x=x, edge_index=edge_index, y=y)

class Lerna(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    @lru_cache(maxsize=1)
    def processed_file_names(self):
        return [os.path.splitext(raw)[0] + '.pt' for raw in self.raw_file_names]

    def __len__(self):
        return len(self.raw_file_names)

    def download(self):
        assert False, "download not yet available"

    def process(self):
        for name in tqdm(self.raw_file_names):
            data = load_graph(os.path.join(self.raw_dir, name))

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            root = os.path.splitext(name)[0]
            processed_name = os.path.join(self.processed_dir, root + '.pt')
            torch.save(data, processed_name)

    def get(self, idx):
        name = self.processed_file_names[idx]
        return torch.load(os.path.join(self.processed_dir, name))
