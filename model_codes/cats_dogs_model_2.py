import os
import torch
from torch import nn
from datasets import load_dataset, Image
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter, \
    Resize, ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from helper_functions import save_model
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# HYPERPARAMETERS
BATCH_SIZE = 64
IMG_SIZE = (64, 64)
RAMDOM_SEED = 11
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_WORKERS = os.cpu_count()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Augmentations
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=(-10, 10)),
    RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor()
])

test_transform = Compose([
    Resize(IMG_SIZE),
    ToTensor()
])



def main():
    dataset, train_dataloader, test_dataloader = ready_data()
    cats_dogs_model_1 = CatsAndDogsV2(input_shape=3, hidden_units=24, output_shape=1).to(device)

    torch.cuda.manual_seed(RAMDOM_SEED)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=cats_dogs_model_1.parameters(), lr=LEARNING_RATE)

    start_time = timer()
    cats_dogs_model_1_results = train(model=cats_dogs_model_1,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=EPOCHS)
    end_time = timer()
    print(f"\nTotal training time: {end_time - start_time:.3f} seconds")

    plot_loss_curves(cats_dogs_model_1_results)
    save_model(name='cats_dogs_model_1.pth', model=cats_dogs_model_1)


def ready_data():
    dataset = load_dataset("Bingsu/Cat_and_Dog")


    train_dataloader = DataLoader(dataset['train'],
                                  collate_fn=collate_fn,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset['test'],
                                 collate_fn=collate_fn,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

    return dataset, train_dataloader, test_dataloader


class CatsAndDogsV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
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
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch in dataloader:
        X, y = batch['image'], batch['labels']
        X, y = X.to(device), y.to(device)
        y = y.type(torch.float32)

        y_logits = model(X).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (y_preds == y).sum().item() / len(y_preds)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch in dataloader:
            X, y = batch['image'], batch['labels']
            X, y = X.to(device), y.to(device)
            y = y.type(torch.float32)

            test_pred_logits = model(X).squeeze()
            test_pred = torch.round(torch.sigmoid(test_pred_logits))

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_acc += ((test_pred == y).sum().item() / len(test_pred))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          scheduler=None,
          epochs: int = 5):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        if scheduler:
            scheduler.step()

        print(
            f"\n\nEpoch: {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def plot_loss_curves(results):
    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
