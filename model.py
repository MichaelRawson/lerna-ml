import torch
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.nn.functional import dropout, log_softmax, relu
from torch_geometric.nn import GCNConv, TopKPooling, global_max_pool

FLAVOURS = 17
DROPOUT = 0.1

class Conv(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = GCNConv(channels, channels)

    def forward(self, x, edge_index, batch):
        x = dropout(x, p=DROPOUT, training=self.training)
        x = self.conv(x, edge_index)
        return x, edge_index, batch

class Pool(Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.pool = TopKPooling(channels, ratio)

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _ = self.pool(x, edge_index, batch=batch)
        return x, edge_index, batch

class FC(Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.linear = Linear(channels_in, channels_out)
        
    def forward(self, x):
        x = dropout(x, p=DROPOUT, training=self.training)
        x = self.linear(x)
        return x

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(FLAVOURS, 64)
        self.initial_conv = ModuleList([Conv(64) for _ in range(4)])
        self.conv = ModuleList([Conv(64) for _ in range(3)])
        self.pool = ModuleList([Pool(64, 0.6) for _ in range(3)])
        self.fc0 = FC(64, 64)
        self.fc1 = FC(64, 32)
        self.fc2 = FC(32, 2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        for conv in self.initial_conv:
            x, edge_index, batch = conv(x, edge_index, batch)
            x = relu(x)

        for conv, pool in zip(self.conv, self.pool):
            x, edge_index, batch = conv(x, edge_index, batch)
            x = relu(x)
            x, edge_index, batch = pool(x, edge_index, batch)

        x = global_max_pool(x, batch)
        x = self.fc0(x)
        x = relu(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = log_softmax(x, dim=1)

        return x
