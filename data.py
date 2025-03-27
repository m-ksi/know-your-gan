import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import random

def get_dataset(conf):
    match conf['type']:
        case 'CelebA' | 'CelebA32':
            cl = CelebA
        case 'CIFAR10':
            cl = Cifar10
    return cl(**conf['params'])

class CelebA(Dataset):
    def __init__(self, imsize=32, train_aug=True):
        if imsize == 128:
            self.images = os.listdir('datasets/CelebA/CelebA-img')
        elif imsize == 32:
            self.images = os.listdir('datasets/CelebA32/CelebA-img')
        self.images = [i for i in self.images if int(i.split('.')[0]) < 162771] # only train images
        self.imsize = imsize
        self.train_aug = train_aug
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join('datasets/CelebA/CelebA-img', self.images[index])))
        img = self.transform(img)
        if self.train_aug and random.random() < 0.5:
            img = torch.flip(img, [2])
        if self.imsize != img.shape[1]:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), (self.imsize, self.imsize), mode='area')[0]
        return img
        
class Cifar10(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.ds = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]
