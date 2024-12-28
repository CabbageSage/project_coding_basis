from diffusers import DDPMScheduler, UNet2DModel
import torch
import torch.nn as nn


class ClassConditionedUnet(nn.Module):
    def __init__(self, image_size, num_classes=10, class_emd_size=4):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emd_size)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels = 1+class_emd_size,
            out_channels = 1,
            layers_per_block = 2,
            block_out_channels = (32, 64, 64),
            down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )
        
    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, -1, 1, 1).repeat(1, 1, w, h)
        x = torch.cat([x, class_cond], dim=1)
        return self.model(x, t).sample
    
    
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")