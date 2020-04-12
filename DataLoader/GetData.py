
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensor
from albumentations import Compose, RandomCrop,PadIfNeeded,Flip, Normalize, HorizontalFlip, Cutout, VerticalFlip, Rotate, RGBShift
from albumentations.pytorch import ToTensor

class albumCompose:
    def __init__(self):
        self.albumentations_transform = Compose({
            PadIfNeeded(min_height=40,min_width=40),
            RandomCrop(32,32),
            Flip(),
            VerticalFlip(p=0.5),
            Cutout(max_h_size=8,max_w_size=8,num_holes=1),
            Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        })
    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img, dtype=torch.float)

class albumCompose_test:
    def __init__(self):
        self.albumentations_transform = Compose({
        Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        })
    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img, dtype=torch.float)


class GetData():
    def importDataset():
        SEED = 1
        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        # For reproducibility
        device = torch.device("cuda" if cuda else "cpu")
        
        torch.manual_seed(SEED)
        
        if cuda:
            torch.cuda.manual_seed(SEED)

        # train_transforms = transforms.Compose([
        #                                 transforms.RandomCrop(32, padding=4),
        #                                 transforms.RandomHorizontalFlip(),
        #                                 transforms.Cutout()
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        #                                ])
        # test_transforms = transforms.Compose([
        #                                transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #                                ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=albumCompose())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=albumCompose_test())
        testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=4)
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader, classes, device
