import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

_PATH = "./data/MNIST/preprocessed/"
_file_path = lambda x: os.path.join(_PATH, x)


def preprocess():
    os.makedirs(_PATH, exist_ok=True)

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = MNIST("./data/", train=True, download=True, transform=transform)
    test_set = MNIST("./data/", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))

    torch.save(train_data, _file_path("train.pt"))
    torch.save(test_data, _file_path("test.pt"))


if __name__ == "__main__":
    preprocess()

assert os.path.exists(_PATH), f"Preprocessed data not found. Run: python {__file__}"

train_x, train_y = torch.load(_file_path("train.pt"))
test_x, test_y = torch.load(_file_path("test.pt"))
