import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from helper_functions import accuracy_fn, plot_decision_boundary, save_model

#Hyperparamaters
EPOCHS = 1000

LEARNING_RATE = .1

#00 Getting Data Ready
NUM_SAMPLES = 1000
NUM_CLASSES = 4
NUM_FEATURES = 2  #number of inputs
RANDOM_SEED = 11

x_blob, y_blob = make_blobs(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob, test_size=0.2,
                                                                        random_state=RANDOM_SEED)

plt.figure(figsize=(10, 7))
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

#01 Building Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


multi_class_model_0 = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES).to(device)

#02 Loss and Optimization Functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=multi_class_model_0.parameters(), lr=LEARNING_RATE)


#03 Training and Test Function
def train_test_step(model: torch.nn.Module,
                    x_train: torch.float32, y_train: torch.float32,
                    x_test: torch.float32, y_test: torch.float32,
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    model.train()

    y_logits = model(x_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 100 == 0:
        print(
            f'\nEpoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.5f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}\n%')


#04 Fitting the Model
x_blob_train, y_blob_train = x_blob_train.to(device), y_blob_train.to(device)
x_blob_test, y_blob_test = x_blob_test.to(device), y_blob_test.to(device)

for epoch in tqdm(range(EPOCHS)):
    train_test_step(model=multi_class_model_0, x_train=x_blob_train, y_train=y_blob_train, x_test=x_blob_test,
                    y_test=y_blob_test, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch)


#05 Making Predictions and Evaluation
multi_class_model_0.eval()
with torch.inference_mode():
    y_logits = multi_class_model_0(x_blob_test.to(device))
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

print(f'\nPredictions: {y_preds[:10]}, True Values: {y_blob_test[:10]}')

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(multi_class_model_0, x_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(multi_class_model_0, x_blob_test, y_blob_test)
plt.show()

#06 Saving Model
save_model(name='multi_class_model_0.pth', model=multi_class_model_0)
