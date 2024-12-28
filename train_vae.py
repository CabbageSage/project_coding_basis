import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from data_loader import get_loader
from utils.utils import plot_loss, set_seed
from Models.vae import VAE
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

import os
import sys

# parameters
import config
opt = config.get_options()
batch_size = opt.batch_size
image_size = opt.image_size
epochs = opt.epochs
lr = opt.lr
num_workers = opt.workers
manual_seed = opt.seed
latent_dim = 20
device = 'cuda' if opt.cuda else 'cpu'

model = VAE(image_size).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=lr)

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = criterion(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# train and test
train_losses = []
test_losses = []

#train
train_loader = get_loader(image_size, batch_size, num_workers, train=True)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data)
        loss = vae_loss(recon_x, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")
    train_losses.append(train_loss / len(train_loader.dataset))
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

# test
test_loader = get_loader(image_size, batch_size, num_workers, train=False)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            recon_x, mu, logvar = model(data)
            loss = vae_loss(recon_x, data, mu, logvar)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"====> Test set loss: {test_loss:.4f}")


if __name__ == "__main__":
        
    # test whether the dir exists
    results_dir = './results/vae'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        if os.listdir(results_dir):
            raise Exception(f"The {results_dir} directory is not empty.")

    set_seed(manual_seed)
    
    for epoch in tqdm(range(1, epochs + 1)):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim).to(device)
            sample = model.decode(z).view(batch_size, 1, image_size, image_size)
            save_image(sample, f'{results_dir}/vae_samples_epoch_{epoch}.png')
    # save model
    checkpoint_path = './checkpoints'
    torch.save(model.state_dict(),os.path.join(checkpoint_path, 'vae.pth'))

    #save loss
    with open(os.path.join(results_dir, 'vae_loss.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
            writer.writerow([epoch, train_loss, test_loss])

    # save loss figure
    plot_loss(train_losses, test_losses, results_dir)