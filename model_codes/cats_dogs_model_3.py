import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from pathlib import Path
from helper_functions import save_model, plot_loss_curves
import data_setup
import engine
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

#HYPERPARAMATERS
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
RANDOM_SEED = 11
EPOCHS = 2

#Augmentation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def ready_data():
    MNIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CatsVsDogs')
    MNIST_PATH = Path(os.path.abspath(MNIST_PATH))

    # Setup Dirs
    train_dir = MNIST_PATH / "training_set"
    test_dir = MNIST_PATH / "test_set"

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=transform,
                                                                                   batch_size=BATCH_SIZE)

    return train_dataloader, test_dataloader, class_names


if __name__ == '__main__':
    train_dataloader, test_dataloader, class_names = ready_data()
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT                   #Use EfficientNet_B0 CNN Model
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False  #Freeze model first then adjust the output_features

    torch.cuda.manual_seed(RANDOM_SEED)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=1,  # same number of output units as our number of classes (for binary set to 1)
                        bias=True)).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = timer()
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=EPOCHS,
                           device=device)

    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    plot_loss_curves(results)
    # save_model(model=model, name='CatsVsDogsV3.pth')


