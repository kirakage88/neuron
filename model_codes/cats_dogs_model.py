import os
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
from tqdm.auto import tqdm
from helper_functions import save_model, plot_loss_curves
from timeit import default_timer as timer

# HYPERPARAMETERS
RANDOM_SEED = 11
IMG_SIZE = (64, 64)
BATCH_SIZE = 128
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    #00 Getting Data Ready
    train_data_simple, test_data_simple, train_dataloader_simple, test_dataloader_simple = ready_data()

    #01 Building Model

    cats_dogs_model_0 = CatsAndDogs(input_shape=3, hidden_units=24, output_shape=1).to(device)

    #02 Fitting Model
    torch.cuda.manual_seed(RANDOM_SEED)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=cats_dogs_model_0.parameters(), lr=LEARNING_RATE)
        #schedule decay every 5 epochs

    start_time = timer()
    cats_dogs_model_0_results = train(model=cats_dogs_model_0,
                                      train_dataloader=train_dataloader_simple,
                                      test_dataloader=test_dataloader_simple,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=NUM_EPOCHS)

    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    plot_loss_curves(cats_dogs_model_0_results)

    # 03 Saving Model
    save_model(name='cats_dogs_model_0.pth', model=cats_dogs_model_0)


def ready_data():
    #00 Getting Data Ready
    MNIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CatsVsDogs')
    MNIST_PATH = Path(os.path.abspath(MNIST_PATH))

    train_dir = MNIST_PATH / "training_set"
    test_dir = MNIST_PATH / "test_set"

    random.seed(RANDOM_SEED)
    image_path_list = list(MNIST_PATH.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)

    img_as_array = np.asarray(img)

    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array, cmap='gray')
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    plt.axis(False)
    plt.show()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    train_data_simple = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=test_transform)

    print(train_data_simple)

    # 01 Turning Loaded Images into DataLoader
    train_dataloader_simple = DataLoader(train_data_simple,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=NUM_WORKERS,
                                         pin_memory=True)

    test_dataloader_simple = DataLoader(test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    return train_data_simple, test_data_simple, train_dataloader_simple, test_dataloader_simple


class CatsAndDogs(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
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
            nn.Linear(in_features=hidden_units*32*32,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        y = y.type(torch.float32)

        # 1. Forward pass
        y_logits = model(X).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches

        # print(y_pred_class.shape)
        train_acc += (y_preds == y).sum().item() / len(y_preds)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            y = y.type(torch.float32)

            # 1. Forward pass
            test_pred_logits = model(X).squeeze()
            test_pred = torch.round(torch.sigmoid(test_pred_logits))

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_acc += ((test_pred == y).sum().item() / len(test_pred))

    # Adjust metrics to get average loss and accuracy per batch
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
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
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

        # 4. Print out what's happening
        print(
            f"\nEpoch: {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def plot_loss_curves(results):
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
