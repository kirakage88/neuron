import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path



def get_data(ax, canvas):
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step = .02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))  # 80/20 for train/test split
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    ax.clear()
    ax.scatter(x=X, y=y, c='b', s=4, alpha=0)
    ax.scatter(x=X_train, y=y_train, c='b', s=4, label='Training Data')
    canvas.draw()

    return X_train, y_train, X_test, y_test


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):  #Forward method to define the computation in the model_codes. The x is the input data.
        return self.layer_1(x)


def start_training():
    linear_model_0 = LinearRegressionModel()

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=linear_model_0.parameters(), lr=0.001)
    return linear_model_0, loss_fn, optimizer


