import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_dataset(conf):
    match conf["type"]:
        case "CelebA" | "CelebA32":
            cl = CelebA
        case "CIFAR10":
            cl = Cifar10
        case "FFHQ":
            cl = FFHQ
    return cl(**conf["params"])


class FFHQ(Dataset):
    name = "FFHQ"

    def __init__(self, imsize=64, train_aug=True, **kwargs):
        if imsize == 64:
            self.images = os.listdir("datasets/FFHQ64")
        elif imsize == 32:
            self.images = os.listdir("datasets/FFHQ32")
        self.imsize = imsize
        self.train_aug = train_aug
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join("datasets/FFHQ64", self.images[index])))
        img = self.transform(img)
        if self.train_aug and random.random() < 0.5:
            img = torch.flip(img, [2])
        if self.imsize != img.shape[1]:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), (self.imsize, self.imsize), mode="area"
            )[0]
        return img


class CelebA(Dataset):
    name = "CelebA"

    def __init__(self, imsize=32, train_aug=True, **kwargs):
        if imsize == 128:
            self.images = os.listdir("datasets/CelebA/CelebA-img")
        elif imsize == 32:
            self.images = os.listdir("datasets/CelebA32/CelebA-img")
        self.images = [
            i for i in self.images if int(i.split(".")[0]) < 162771
        ]  # only train images
        self.imsize = imsize
        self.train_aug = train_aug
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(
            Image.open(os.path.join("datasets/CelebA32/CelebA-img", self.images[index]))
        )
        img = self.transform(img)
        if self.train_aug and random.random() < 0.5:
            img = torch.flip(img, [2])
        if self.imsize != img.shape[1]:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), (self.imsize, self.imsize), mode="area"
            )[0]
        return img


class Cifar10(Dataset):
    name = "CIFAR10"

    def __init__(self, **Kwargs):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.ds = datasets.CIFAR10(
            root="./datasets", train=True, download=True, transform=transform
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index][0]
