from torchvision import datasets, transforms
import torch

def get_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return datasets.CIFAR10(root='./data', transform=transform, download=True)