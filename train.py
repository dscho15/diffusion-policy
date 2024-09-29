import torch.amp
from models.vision_encoder import VisionEncoder
from models.conditional_unet_1d import ConditionalUnet1D
from torch.utils.data import DataLoader
from dataset import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm

dataset = Dataset("test")

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

dim_actions = 3
dim_global_conditioning = 512

pred_horizon = 1
obs_horizon = 1
action_horizon = 1
num_epochs = 250

vision_encoder = VisionEncoder().cuda()

cond_unet = ConditionalUnet1D(dim_actions,
                              dim_global_conditioning=dim_global_conditioning * pred_horizon).cuda()

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': cond_unet
})

ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75
)

optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, 
    weight_decay=1e-6)

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:

    for epoch_idx in tglobal:
        
        epoch_loss = list()

        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            
            for (x, naction) in (tepoch):

                x = x.cuda()
                naction = naction.cuda().unsqueeze(1)

                y_pred = nets["vision_encoder"](x)

                b, _ = y_pred.shape

                # sample noise to add to actions
                noise = torch.randn((b, 1, 3), device=y_pred.device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (b,), 
                    device=y_pred.device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                noisy_actions = noise_scheduler.add_noise(
                    naction, 
                    noise, 
                    timesteps
                )

                # predict the noise residual
                noise_pred = nets["noise_pred_net"](
                    noisy_actions, 
                    timesteps, 
                    global_cond=y_pred
                )

                # l2-loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
ema_nets = nets
ema.copy_to(ema_nets.parameters())

# save network
torch.save(ema_nets, "ema_nets.pth")