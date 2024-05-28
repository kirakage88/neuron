import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from helper_functions import accuracy_fn, plot_decision_boundary, save_model


def get_data(ax, canvas):
    x_blob, y_blob = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=1.5,
                                random_state=11)
    x_blob = torch.from_numpy(x_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
    x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob, test_size=0.2,
                                                                            random_state=11)
    ax.clear()
    ax.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
    canvas.draw()

    return x_blob_train, y_blob_train, x_blob_test, y_blob_test


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_untis=8):
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


def start_training():
    blob_model = BlobModel(input_features=2, output_features=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=blob_model.parameters(), lr=.1)

    return blob_model, loss_fn, optimizer
