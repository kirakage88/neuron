import loader
import torch
from helper_functions import predict_image_efficient_net

model = loader.cats_dogs_model_3
IMG_PATH = 'data/preds/sigma2.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_image_efficient_net(model=model, image_path=IMG_PATH, class_names=['Cat', 'Dog'], device=device)
