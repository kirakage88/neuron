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

# HYPERPARAMETERS
IMG_SIZE = (380, 380)              # DO NOT CHANGE
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
RANDOM_SEED = 11
EPOCHS = 20
scheduler = None
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def ready_data():
    MNIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data/WasteClass')
    MNIST_PATH = Path(os.path.abspath(MNIST_PATH))

    train_dir = MNIST_PATH / "train"
    test_dir = MNIST_PATH / "test"

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=transform,
                                                                                   batch_size=BATCH_SIZE)

    return train_dataloader, test_dataloader, class_names


if __name__ == '__main__':
    train_dataloader, test_dataloader, class_names = ready_data()
    print(class_names)

    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model = torchvision.models.efficientnet_b4(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    torch.cuda.manual_seed(RANDOM_SEED)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1792,
                        out_features=3,  # same number of output units as our number of classes (for binary set to 1)
                        bias=True)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

    start_time = timer()
    results = engine.train_multi(model=model,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 epochs=EPOCHS,
                                 device=device,
                                 scheduler=scheduler)

    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    plot_loss_curves(results)
    save_model(model=model, name='waste_class_model_3.pth')
