
from pathlib import Path
import torch
from PIL import Image

import torchvision.transforms as transforms
import random

from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

def to_tensor(x: torch.Tensor):
    x = transforms.ToTensor()(x)
    return x

def do_resize(x: torch.Tensor, size: int = 224):
    x = F.resize(x, 2 * (size, ), interpolation=InterpolationMode.BILINEAR)
    return x

def normalize_action(action, max_rotation=15, max_translation=15):
    rotation = action[0] / max_rotation
    translation_x = action[1][0] / max_translation
    translation_y = action[1][1] / max_translation
    return (rotation, translation_x, translation_y)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, p_data = "imgs"):
        self.data = sorted(list(Path(p_data).rglob("*.jpg")))

    def __len__(self):
        return len(self.data)
    
    def rotate_image(self, x):
        x = F.pad(x, padding=50, padding_mode='reflect')
        action = (random.randint(-15, 15), (random.randint(-15, 15), random.randint(-15, 15)), 1.0, 0.0)

        x = F.affine(x, *action)
        x = F.center_crop(x, output_size=(224, 224))

        action = normalize_action(action)
        action = torch.tensor(action, dtype=torch.float32)

        return x, action
    
    def __getitem__(self, index):
        
        x = Image.open(self.data[index])

        x = to_tensor(x)
        x = do_resize(x)

        x, action = self.rotate_image(x)

        return (x, action)
    
save_image(Dataset("imgs")[0][0], "test.png")