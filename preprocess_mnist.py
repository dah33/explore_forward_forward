import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import os

save_path = './data/MNIST/preprocessed/'
os.makedirs(save_path, exist_ok=True)
file_path = lambda x: os.path.join(save_path, x)

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
    Lambda(lambda x: torch.flatten(x))])

train_set = MNIST('./data/', train=True, download=True, transform=transform)
test_set = MNIST('./data/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

train_x, train_y = next(iter(train_loader))
test_x, test_y = next(iter(test_loader))

torch.save(train_x, file_path('train_x.pt'))
torch.save(train_y, file_path('train_y.pt'))
torch.save(test_x, file_path('test_x.pt'))
torch.save(test_y, file_path('test_y.pt'))
