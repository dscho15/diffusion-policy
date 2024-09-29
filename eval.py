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
    num_workers=4,
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

ema_nets = torch.load("ema_nets.pth")

vision_encoder = VisionEncoder()
vision_encoder.load_state_dict(ema_nets['vision_encoder'].state_dict())
vision_encoder.eval().cuda()

cond_unet = ConditionalUnet1D(dim_actions,
                              dim_global_conditioning=dim_global_conditioning * pred_horizon)
cond_unet.load_state_dict(ema_nets['noise_pred_net'].state_dict())
cond_unet.eval().cuda()

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': cond_unet
})

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
).cuda()

with torch.inference_mode():

    for (x, naction) in (dataloader):

        x = x.cuda()
        action_gt = naction.cuda()

        y_pred = nets["vision_encoder"](x)

        b, _ = y_pred.shape

        # initialize action from Guassian noise
        naction = torch.randn((1, 1, 3), device=y_pred.device)

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        if len(naction.shape) != 3:
            naction = naction.unsqueeze(1)

        for k in noise_scheduler.timesteps:

            # predict noise
            noise_pred = nets['noise_pred_net'](
                sample=naction,
                timestep=k.cuda(),
                global_cond=y_pred
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        
        # unnormalize action
        naction = naction.detach().to('cpu').numpy() * 15 + 15
        print(naction, action_gt * 15 + 15)
