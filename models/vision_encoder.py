import torch
from torch import nn

def replace_batchnorm_with_groupnorm(module, num_groups=32):
    """
    Recursively replaces all instances of BatchNorm with GroupNorm in a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to modify.
        num_groups (int): Number of groups for GroupNorm.
    """
    for name, child in module.named_children():
        replace_batchnorm_with_groupnorm(child, num_groups)
        
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        elif isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))

class VisionEncoder(torch.nn.Module):
        
        def __init__(self):
            super(VisionEncoder, self).__init__()

            self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.resnet.fc = torch.nn.Linear(512, 10)
            self.resnet.fc = torch.nn.Identity()

            replace_batchnorm_with_groupnorm(self.resnet)
            
        def forward(self, x):
            return self.resnet(x)