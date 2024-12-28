import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from PIL import Image
# from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

# import config

# opt = config.get_options()
# batch_size = opt.batch_size
# image_size = 64
latent_dim = 20

class VAE(nn.Module):
    def __init__(self, image_size, latent_dim=latent_dim):
        self.input_dim = image_size * image_size
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, self.latent_dim)
        self.fc_logvar = nn.Linear(400, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, self.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# if __name__ == "__main__":
#     for epoch in range(1, epochs + 1):
#         train(epoch, train_losses)
#         test(epoch, test_losses)
        
#         # 保存生成的图像
#         if not os.path.exists('./pics'):
#             os.makedirs('./pics')
#         if (epoch + 1) % 5 == 0:
#             with torch.no_grad():
#                 z = torch.randn(batch_size, latent_dim).to(device)
#                 sample = model.decode(z).view(batch_size, 1, 64, 64)
#                 save_image(sample, f'./pics/vae_samples_epoch_{epoch+1}.png')

#     # 保存模型
#     torch.save(model.state_dict(), 'vae.pth')
