import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
from torchvision import models


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):  #Forward method to define the computation in the model_codes. The x is the input data.
        return self.layer_1(x)


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(  #Method for compling layers into one block or sequence
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),  #Non-linear function for non-linear data
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        return self.layer_stack(x)


class BlobModel(nn.Module):
    def __init__(self, input_features=2, output_features=4, hidden_untis=8):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_untis),
            nn.ReLU(),
            nn.Linear(in_features=hidden_untis, out_features=hidden_untis),
            nn.ReLU(),
            nn.Linear(in_features=hidden_untis, out_features=output_features),
        )

    def forward(self, x):
        return self.layer_stack(x)


class MNIST(nn.Module):
    def __init__(self, input_shape=3, hidden_units=16, output_shape=10) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


class EfficientNetB0Binary(nn.Module):
    def __init__(self):
        super(EfficientNetB0Binary, self).__init__()
        self.model = models.efficientnet_b0(weights=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 1)  # Change the output layer for binary classification

    def forward(self, x):
        return self.model(x)
