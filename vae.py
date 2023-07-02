#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
#%%
data_dir = Path('data')
batch_size = 32
epochs = 50
image_size = 256
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
full_dataset = torchvision.datasets.ImageFolder(root = data_dir)
train_size = int(len(full_dataset)*0.8)
test_size = int(len(full_dataset) - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_transform = transforms.Compose(
    [
        transforms.Resize(size = image_size),
        transforms.RandomCrop(size = (image_size,image_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(size = image_size),
        transforms.CenterCrop(size = (image_size,image_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
)

train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
#%%
def imshow(img):
    """function to show image"""
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

images,labels = next(iter(train_loader))
imshow(torchvision.utils.make_grid(images))


# %%
