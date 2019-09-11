#!/usr/bin/env python3

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import accuracy, true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from lerna_dataset import Lerna

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST = Lerna('/media/michael/data/lerna-test')

def evaluate(model):
    model.eval()
    loader = DataLoader(TEST, batch_size=BATCH_SIZE)

    pred = []
    target = []
    for data in loader:
        data = data.to(DEVICE)
        predicted = torch.argmax(model(data.x, data.edge_index, data.batch), dim=1)
        for p in predicted:
            pred.append(p.item())
        for y in data.y:
            target.append(y.item())

    pred = torch.tensor(pred)
    target = torch.tensor(target)
    print("Accuracy: {:.2f}%".format(100 * accuracy(pred, target)))
    print("True Positive: {}".format(true_positive(pred, target, 1).item()))
    print("True Negative: {}".format(true_negative(pred, target, 1).item()))
    print("False Positive: {}".format(false_positive(pred, target, 1).item()))
    print("False Negative: {}".format(false_negative(pred, target, 1).item()))
    print("Precision: {:.2f}%".format(100 * precision(pred, target, 1).item()))
    print("Recall: {:.2f}%".format(100 * recall(pred, target, 1).item()))
    print("F1 score: {:.2f}%".format(100 * f1_score(pred, target, 1).item()))

if __name__ == '__main__':
    model = torch.load('model.pt')
    evaluate(model)
