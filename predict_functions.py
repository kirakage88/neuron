import torch
import numpy as np

from torch import nn

import os
import zipfile
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import model_class
from typing import List, Tuple
from PIL import Image

from torchvision import models, transforms

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os


def predict_image_mnist(model: torch.nn.Module,
                        image_path: str,
                        class_names=None,
                        image_size: Tuple[int, int] = (28, 28),
                        transform: transforms = None,
                        device=None):
    if class_names is None:
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    img = Image.open(image_path)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_logits = model(transformed_image.to(device)).squeeze(dim=0)

        target_image_pred_probs = target_image_logits.softmax(dim=0)
        target_image_label = target_image_pred_probs.argmax(dim=0)

        target_image_pred_probs *= 100


    return img, class_names[target_image_label], target_image_pred_probs.max()


def predict_image_efficient_net(model: torch.nn.Module,
                                image_path: str,
                                class_names: List[str]= ['Cat', 'Dog'],
                                image_size: Tuple[int, int] = (224, 224),
                                transform: transforms = None,
                                device=None):
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_logits = model(transformed_image.to(device)).squeeze(dim=0)


    target_image_pred_probs = torch.sigmoid(target_image_logits)
    print(target_image_pred_probs)

    target_image_pred_label = int(torch.round(target_image_pred_probs).item())

    if target_image_pred_label == 0:
        target_image_pred_probs = 1 - target_image_pred_probs

    target_image_pred_probs *= 100

    return img, class_names[target_image_pred_label], target_image_pred_probs.item()