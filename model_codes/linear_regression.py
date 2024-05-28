import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path

# <editor-fold desc="Initializing Data">
weight = 0.7
bias = 0.3
start = 0
end = 1
step = .02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create a train/test split

train_split = int(0.8 * len(X))  # 80/20 for train/test split
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# </editor-fold>


epoch_count = []
loss_values = []
test_loss_values = []


def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):  #Forward method to define the computation in the model_codes. The x is the input data.
        return self.layer_1(x)


#Setting Model
torch.manual_seed(42)
linear_model_0 = LinearRegressionModel()

#Loss and Optimization Functions
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=linear_model_0.parameters(), lr=0.001)


#Training and Testing Functions
def train_step(model: torch.nn.Module,
               x,
               y,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int):
    model.train()
    # 1. Forward pass
    y_pred = model(x)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} \nTrain Loss: {loss}')
        loss = torch.tensor(loss).numpy()
        loss_values.append(loss)


def test_step(model: torch.nn.Module,
              x,
              y,
              loss_fn: torch.nn.Module,
              epoch: int):
    model.eval()  # put model_codes in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model(x)

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_pred, y)

        # Adjust metrics and print out
        if epoch % 100 == 0:
            print(f'Test Loss: {test_loss}')
            epoch_count.append(epoch)
            test_loss = torch.tensor(test_loss).numpy()
            test_loss_values.append(test_loss)


#Training Model
epochs = 2100

torch.manual_seed(42)

train_start_time = timer()
for epoch in tqdm(range(epochs + 1)):
    train_step(linear_model_0, x=X_train, y=y_train, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch)
    test_step(linear_model_0, x=X_test, y=y_test, loss_fn=loss_fn, epoch=epoch)

#Eval Model
torch.manual_seed(42)


def eval_model(model: torch.nn.Module,
               x,
               y,
               loss_fn: torch.nn.Module):
    model.eval()  # put model_codes in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model(x)

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_pred, y)

    return {"model_name": model.__class__.__name__,  # only works when model_codes was created with a class
            "model_loss": test_loss.item()}


#Get Model Results
linear_model_0_results = eval_model(
    model=linear_model_0,
    x=X_test,
    y=y_test,
    loss_fn=loss_fn
)
print(linear_model_0_results)

MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(parents=True,  # create parent directories if needed
                 exist_ok=True  # if models directory already exists, don't error
                 )

# Create model_codes save path
MODEL_NAME = "linear_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model_codes state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=linear_model_0.state_dict(),  # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

#Visualize Model Predictions
with torch.no_grad():
    test_pred = linear_model_0(X_test)
plot_predictions(predictions=test_pred)

plt.plot(epoch_count, loss_values, label='Train Loss')
plt.plot(epoch_count, test_loss_values, label='Test Loss')
plt.title('Training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
