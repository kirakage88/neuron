#Imports
import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from helper_functions import accuracy_fn, plot_decision_boundary, save_model
from tqdm.auto import tqdm
from pathlib import Path


#Hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.3

#Creating a Dataset
N_SAMPLES = 1000
RANDOM_STATE = 11

x, y = make_circles(N_SAMPLES, noise=0.03, random_state=RANDOM_STATE)

#00 Getting Data Ready
##Visualize Dataset
circles = pd.DataFrame({'X1': x[:, 0], 'X2': x[:, 1], 'label': y})
print(circles)
plt.scatter(x=circles['X1'], y=circles['X2'], c=circles['label'], cmap=plt.cm.RdYlBu)
plt.show()

##Turning data into tensors
x = torch.from_numpy(x).type(torch.float)  #float64 to float32
y = torch.from_numpy(y).type(torch.float)

##Split the data into training and test sets (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=RANDOM_STATE)  #test_size in percentage
print(len(x_test), len(y_test))

#01 Building a Model
##Device Agnostic Code
device = 'cuda' if torch.cuda.is_available() else 'cpu'  #change to gpu in google colab runtime settings
print(device)


##Binary Model
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


binary_model_0 = CircleModel().to(device)
print(binary_model_0)

#03 Setup Loss and Optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(binary_model_0.parameters(), lr=LEARNING_RATE)


##04 Training and Testing Functions
def train_test_step(model: torch.nn.Module,
                    x_train: torch.float32, y_train: torch.float32,
                    x_test: torch.float32, y_test: torch.float32,
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    model.train()

    y_logits = model(x_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 100 == 0:
        print(f'\nEpoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.5f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}\n%')


x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)


for epoch in tqdm(range(EPOCHS)):
    train_test_step(model=binary_model_0, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch)

#05 Make Predictions
binary_model_0.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(binary_model_0(x_test))).squeeze()
print(y_preds[:10], y_test[:10])

##Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(binary_model_0, x_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(binary_model_0, x_test, y_test) # model_3 = has non-linearity
plt.show()

#06 Saving Model
save_model(name='binary_model_0.pth', model=binary_model_0)



