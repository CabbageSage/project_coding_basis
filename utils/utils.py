import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os

def plot_loss(train_losses, test_losses, path):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training loss', color='red') 
    plt.plot(test_losses, label='Test loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, '/vae_loss.png'))
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
