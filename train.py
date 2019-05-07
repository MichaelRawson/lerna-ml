#!/usr/bin/env python3

from lerna_dataset import Lerna
from model import Model
import torch
from torch.nn.functional import nll_loss
from torch_geometric.data import DataLoader
from tqdm import tqdm
from itertools import islice

PATH = '/media/michael/data/lerna-ml/'
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
EPOCHS = 200

def epoch(optimizer, model, data):
    model.train()
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    losses = []

    loader = tqdm(loader)
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        predicted = model(data.x, data.edge_index, data.batch)
        loss = nll_loss(predicted, data.y)
        loss.backward()
        loss = loss.item()
        losses.append(loss)
        optimizer.step()
        rolling_losses = losses[-50:]
        rolling_loss = sum(rolling_losses) / len(rolling_losses)
        loader.set_postfix({"NLL loss": rolling_loss}, refresh=False)

    return sum(losses) / len(losses)

def train(data):
    best_loss = 100000
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(EPOCHS):
        loss = epoch(optimizer, model, data)
        print(f"{i + 1},{loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            torch.save(model, 'model.pt')

if __name__ == '__main__':
    torch.manual_seed(SEED)
    data = Lerna(PATH)
    train(data)
