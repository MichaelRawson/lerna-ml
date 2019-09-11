from functools import lru_cache
import json
import torch
from tqdm import tqdm
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected

def load_graph(path):
    with open(path, 'rb') as f:
        record = json.load(f)
        x = torch.tensor(record['nodes'], dtype=torch.long)
        edge_index = to_undirected(torch.tensor(record['edges'], dtype=torch.long).t(), num_nodes=len(x))
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
            root = os.path.splitext(name)[0]
            raw_name = os.path.join(self.raw_dir, name)
            processed_name = os.path.join(self.processed_dir, root + '.pt')
            if os.path.exists(processed_name):
                continue

            data = None
            try:
                data = load_graph(raw_name)
            except:
                print(f"{raw_name} invalid:")
                raise

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            torch.save(data, processed_name)

    def get(self, idx):
        processed_name = os.path.join(self.processed_dir, self.processed_file_names[idx])
        try:
            return torch.load(processed_name)
        except:
            print(f"{processed_name} invalid:")
            raise
