import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
from helper_functions import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH_LINEAR = 'models/linear_model_0.pth'
PATH_BINARY = 'models/binary_model_0.pth'
PATH_MULTI = 'models/multi_class_model_0.pth'
PATH_MNIST = 'models/mnist_model_0.pth'
PATH_CVD = 'models/CatsVsDogsV3.pth'
PATH_WASTE = 'models/waste_class_model_3.pth'


linear_model_0 = load_model(model='linear', path=PATH_LINEAR).to(device)
binary_model_0 = load_model(model='binary', path=PATH_BINARY).to(device)
multi_class_model_0 = load_model(model='multi', path=PATH_MULTI).to(device)
mnist_model_0 = load_model(model='mnist', path=PATH_MNIST).to(device)
cats_dogs_model_3 = load_model(model='CVD', path=PATH_CVD).to(device)
waste_model_3 = load_model(model='waste', path=PATH_WASTE).to(device)













