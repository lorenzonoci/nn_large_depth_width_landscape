import torch.nn as nn
import torch
from torch.nn.init import _calculate_correct_fan, calculate_gain
import math 
from torch.nn import init

class LayerNorm2d(nn.Module):
    def __init__(self, nchan):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(nchan))
        self.bias = nn.Parameter(torch.zeros(nchan))

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / x.std(1, keepdim=True, unbiased=False)
        x = x * self.weight.view(1, -1, 1, 1)
        x = x + self.bias.reshape(1, -1, 1, 1)
        return x
 
class MyIdentity(nn.Module):
    def __init__(self):
        super(MyIdentity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x
       
def get_norm_layer(width, norm):
    if norm == None:
        return MyIdentity()
    elif norm == "ln":
        return LayerNorm2d(width)
    elif norm == "bn":
        return nn.BatchNorm2d(width)
    else:
        raise ValueError("Wrong value for normalization layer")
    
    
def get_width_scaling(tensor, nonlinearity='linear', a=1):
    # a is only used for leaky_relu
    fan_in = _calculate_correct_fan(tensor, mode='fan_in') 
    gain = calculate_gain(nonlinearity, a)
    return gain / math.sqrt(fan_in)



class ScaledLayer(nn.Module):
    def __init__(self, layer, sigma_init=1.0, depth_scale=1.0, gamma=1.0, requires_grad=True, nonlinearity='linear'):
        super().__init__()
        self.layer = layer
        self.scaling = get_width_scaling(layer.weight, nonlinearity=nonlinearity) * depth_scale
        self.std_init = sigma_init
        self.gamma = gamma
        self.reset_parameters()
        self._requires_grad(requires_grad)
        
    def forward(self, x):
        return self.scaling/self.gamma * self.layer(x)
    
    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.layer.bias is not None:
                init.zeros_(self.layer.bias)
            self.layer.weight.normal_(0, self.std_init)
            
    def _requires_grad(self, requires_grad):
        self.layer.weight.requires_grad = requires_grad
        if self.layer.bias is not None:
            self.layer.bias.requires_grad = requires_grad
            
            
            
class ScaledResidualBranch(nn.Module):
    def __init__(self, branch, res_scaling=1.0):
        super().__init__()
        self.res_scaling = res_scaling
        self.branch = branch
        
    def forward(self, x):
        return self.res_scaling * self.branch(x)
    
    def set_to_lazy(self):
        self.branch.set_to_lazy()