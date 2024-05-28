import loader
import torch
from helper_functions import predict_image_waste, channel_reader

model = loader.waste_model_3
print(model)
IMG_PATH = 'data/preds/digit_3_normal.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_image_waste(model=model,
                    image_path=IMG_PATH,
                    device=device)

