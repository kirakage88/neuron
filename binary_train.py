import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd


def get_data(ax, canvas):
    x, y = make_circles(1000, noise=0.03, random_state=11)
    circles = pd.DataFrame({'X1': x[:, 0], 'X2': x[:, 1], 'label': y})
    print(circles)

    ax.clear()
    ax.scatter(x=circles['X1'], y=circles['X2'], c=circles['label'], cmap=plt.cm.RdYlBu)
    canvas.draw()

    x = torch.from_numpy(x).type(torch.float)  # float64 to float32
    y = torch.from_numpy(y).type(torch.float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=11)

    return x_train, y_train, x_test, y_test


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        return self.layer_stack(x)


def start_training():
    binary_model = CircleModel()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=binary_model.parameters(), lr=0.3)

    return binary_model, loss_fn, optimizer
