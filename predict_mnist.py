import loader
import torch
from helper_functions import predict_image_mnist, channel_reader

model = loader.mnist_model_0
IMG_PATH = 'data/preds/digit_3_normal.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_image_mnist(model=model,
                    image_path=IMG_PATH,
                    device=device)
