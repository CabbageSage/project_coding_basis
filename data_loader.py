import torch
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def get_loader(image_size, batch_size, num_workers, train=True):
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    data = datasets.MNIST(root='./data', train=train, download=True, transform=data_transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    
    return loader