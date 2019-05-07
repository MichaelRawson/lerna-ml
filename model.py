import torch
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.nn.functional import dropout, embedding, log_softmax, relu
from torch_geometric.nn import GCNConv, global_max_pool

FLAVOURS = 17
CONV_CHANNELS = 32
CONV_LAYERS = 5
DENSE_CHANNELS = 32
DENSE_LAYERS = 3
POOL = 0.9
DROPOUT = 0.1

class Conv(Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = GCNConv(channels, channels)

    def forward(self, x, edge_index, batch):
        x = dropout(x, p=DROPOUT, training=self.training)
        x = self.conv(x, edge_index)
        x = relu(x)
        return x, edge_index, batch

class RectifiedLinear(Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.linear = Linear(channels_in, channels_out)
        
    def forward(self, x):
        x = dropout(x, p=DROPOUT, training=self.training)
        x = self.linear(x)
        x = relu(x)
        return x

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(FLAVOURS, CONV_CHANNELS)
        self.conv = ModuleList([Conv(CONV_CHANNELS) for _ in range(CONV_LAYERS)])
        self.dense0 = RectifiedLinear(CONV_CHANNELS, DENSE_CHANNELS)
        self.dense = ModuleList([RectifiedLinear(DENSE_CHANNELS, DENSE_CHANNELS) for _ in range(DENSE_LAYERS)])
        self.final = Linear(DENSE_CHANNELS, 2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x)
        for conv in self.conv:
            x, edge_index, batch = conv(x, edge_index, batch)

        x = global_max_pool(x, batch)
        x = self.dense0(x)

        for dense in self.dense:
            x = dense(x)

        x = self.final(x)
        x = log_softmax(x, dim=1)
        return x
