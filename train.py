#!/usr/bin/env python3

from pathlib import Path
from lerna_dataset import Lerna
from model import Model
import torch
from torch.nn.functional import nll_loss
from torch_geometric.data import DataLoader
from tqdm import tqdm

ROOT = Path('/media/michael/data/')
TRAIN = Lerna(ROOT / 'lerna-test')#Lerna(ROOT / 'lerna-train')
TEST = Lerna(ROOT / 'lerna-test')

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
EPOCHS = 50

def epoch(optimizer, model):
    model.train()
    loader = DataLoader(TRAIN, batch_size=BATCH_SIZE, shuffle=True)
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

def test(model):
    model.eval()
    loader = DataLoader(TEST, batch_size=BATCH_SIZE)

    losses = []
    for data in loader:
        data = data.to(DEVICE)
        predicted = model(data.x, data.edge_index, data.batch)
        loss = nll_loss(predicted, data.y)
        loss = loss.item()
        losses.append(loss)

    return sum(losses) / len(losses)

def train():
    best_loss = 100000
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(EPOCHS):
        train_loss = epoch(optimizer, model)
        test_loss = test(model)
        print(f"{i + 1},{train_loss:.4f},{test_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, 'model.pt')

if __name__ == '__main__':
    torch.manual_seed(SEED)
    train()
