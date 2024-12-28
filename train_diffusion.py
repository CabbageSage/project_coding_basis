import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from data_loader import get_loader
from utils.utils import plot_loss, set_seed
from Models.diffusion import ClassConditionedUnet, noise_scheduler
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import pickle
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
device = 'cuda' if opt.cuda else 'cpu'

model = ClassConditionedUnet(image_size=image_size).to(device)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loader = get_loader(image_size, batch_size, num_workers, train=True)

losses = []
def train(epoch, losses):
    model.train()
    for x, y in tqdm(train_loader):

        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = model(noisy_x, timesteps, y)  # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise)  # How close is the output to the noise

        # Backprop and update the params:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss for later
        losses.append(loss.item())
        # if batch_idx % 1000 == 0:
        #     print(f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
        #             f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(x):.6f}")

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:]) / 100
    print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

results_dir = './results/diffusion'

if __name__ == '__main__':
    # test whether the dir exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        if os.listdir(results_dir):
            raise Exception("The f{result_dir} directory is not empty.")
        
    # set_seed(manual_seed)    
    
    for epoch in range(1, epochs + 1):
        train(epoch, losses)
    print("Finished training!")
    
    # Save the model
    checkpoint_path = './checkpoints'
    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'diffusion_model.pth'))
    print("Model saved!")
    
    # Save the losses
    with open(os.path.join(results_dir, 'diffusion_losses.pkl'), 'wb') as f:
        pickle.dump(losses, f)