from torch import nn
import torch.nn.functional as F
from architectures.utils import get_norm_layer, ScaledLayer, ScaledResidualBranch
import math
 
 # TODO: handle bias weights
    
class BlockLayer(nn.Module):
    def __init__(self, fan_in, fan_out, norm, kernel_size=3, stride=1, non_lin_first=True, nonlinearity='relu', sigma_init=1.0, bias=None, learnable=True):
        super().__init__()
        if nonlinearity == 'relu':
            nonlin = nn.ReLU()
        else:
            raise ValueError()
        self.sigma_init = sigma_init
        
        conv = ScaledLayer(nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, stride=stride, padding=1, bias=bias), sigma_init=self.sigma_init, requires_grad=learnable, nonlinearity=nonlinearity)
        norm = get_norm_layer(fan_out, norm)
        self.non_lin_first = non_lin_first
        # TODO: check order of operations (especially normalization layers)
        self.layer = nn.ModuleList([nonlin, conv, norm]) if non_lin_first else nn.ModuleList([conv, norm, nonlin])
        
    def forward(self, x):
        for component in self.layer:
            x =  component(x)
        return x
    
    def _requires_grad(self, requires_grad):
        for component in self.layer:
            if isinstance(component, ScaledLayer):
                component._requires_grad(requires_grad)
                

class ResidualBranch(nn.Module):
    def __init__(self, block_class, fan_in, fan_out, n_layers=1, sigma_init_last=1.0, sigma_init=1.0, **kwargs):
        super().__init__()
        self.branch = nn.ModuleList([block_class(fan_in, fan_out, sigma_init=sigma_init, **kwargs) for _ in range(n_layers-1)])
        self.branch.append(block_class(fan_in, fan_out, sigma_init=sigma_init_last, **kwargs))
        
    def forward(self, x):
        for layer in self.branch:
            x = layer(x)
        return x
    
    def set_to_lazy(self):
        if len(self.branch) == 1:
            return
        for layer in self.branch[:-1]:
            layer._requires_grad(False)
    
# def res_branch_factory(block_class, n, width, layers_per_block, sigma_last_layer_per_block, norm, non_lin_first):
#     if n==0:
#         return block_class
#     else:
#         return ResidualBranch(res_branch_factory(block_class, n-1), width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first)
                  
# class FConvNet(nn.Module):
#     def __init__(self, width, n_blocks, rec_level=1, res_scaling=1, skip_scaling=1, beta=1, gamma_zero=1, num_classes=10, img_dim=32,
#                  norm=None, layers_per_block=1, zero_init_readout=False, non_lin_first=True, gamma='sqrt_width', init_stride=1,
#                  depth_scale_non_res_layers=False, sigma_last_layer_per_block=1, bias=None):
#         super().__init__()

#         self.tot_n_blocks = n_blocks
#         self.n_blocks = n_blocks // 3
#         self.res_scaling = res_scaling
#         self.skip_scaling = skip_scaling
#         self.beta = beta
#         self.gamma_zero = gamma_zero
#         self.img_dim = img_dim
#         self.zero_init_readout = zero_init_readout
#         self.depth_scale_non_res_layers = depth_scale_non_res_layers
#         self.rec_level = rec_level
        
#         depth_scale = 1.0 if not self.depth_scale_non_res_layers else math.sqrt(self.tot_n_blocks)

#         self.conv01 = ScaledLayer(nn.Conv2d(3, width, 3, stride=init_stride, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)

#         self.block1 = nn.ModuleList([ScaledResidualBranch(res_branch_factory(BlockLayer, rec_level, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         self.conv02 = ScaledLayer(nn.Conv2d(width, 2*width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)
        
#         self.block2 = nn.ModuleList([ScaledResidualBranch(res_branch_factory(BlockLayer, rec_level, 2*width, 2*width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         self.conv03 = ScaledLayer(nn.Conv2d(2*width, 4*width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)

#         self.block3 = nn.ModuleList([ScaledResidualBranch(res_branch_factory(BlockLayer, rec_level, 4*width, 4*width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         final_size =  self.img_dim//(8*init_stride) 
#         self.final_width = int(final_size**2 * 4 * width)
#         self.gamma = math.sqrt(self.final_width) if gamma == "sqrt_width" else 1.0
        
#         sigma_init = 0.0 if self.zero_init_readout else 1.0
#         self.fc = ScaledLayer(nn.Linear(self.final_width, num_classes, bias=bias), sigma_init=sigma_init*depth_scale, depth_scale=1/depth_scale, gamma=self.gamma)
                
#     def forward(self, x):
#         x = self.conv01(x)
#         for layer in self.block1:
#             x = self.skip_scaling*x + self.beta * layer(x)
#         x = F.max_pool2d(x, 2, 2) 
#         x = self.conv02(x)
#         for layer in self.block2:
#             x =  self.skip_scaling*x + self.beta * layer(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = self.conv03(x)
#         for layer in self.block3:
#             x =  self.skip_scaling*x + self.beta * layer(x)
#         x = F.max_pool2d(x, 2, 2)

#         x = x.view(-1, self.final_width)
#         x = self.fc(x)
        
#         return x
    
#     def set_to_lazy(self):
#         for branch in self.block1:
#             branch.set_to_lazy()
#         for branch in self.block2:
#             branch.set_to_lazy()
#         for branch in self.block3:
#             branch.set_to_lazy()
            
            
# class ConvNet(nn.Module):
#     def __init__(self, width, n_blocks, res_scaling=1, skip_scaling=1, beta=1, gamma_zero=1, num_classes=10, img_dim=32,
#                  norm=None, layers_per_block=1, zero_init_readout=False, non_lin_first=True, gamma='sqrt_width', init_stride=1,
#                  depth_scale_non_res_layers=False, sigma_last_layer_per_block=1, bias=None, base_width=1.0, sigma_init=1.0):
#         super().__init__()

#         self.tot_n_blocks = n_blocks
#         self.n_blocks = n_blocks // 3
#         self.res_scaling = res_scaling
#         self.skip_scaling = skip_scaling
#         self.beta = beta
#         self.gamma_zero = gamma_zero
#         self.img_dim = img_dim
#         self.zero_init_readout = zero_init_readout
#         self.depth_scale_non_res_layers = depth_scale_non_res_layers
        
#         depth_scale = 1.0 if not self.depth_scale_non_res_layers else math.sqrt(self.tot_n_blocks)

#         self.conv01 = ScaledLayer(nn.Conv2d(3, width, 3, stride=2*init_stride, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale*sigma_init, base_width=base_width)
        
#         self.block1 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first, base_width=base_width, sigma_init=sigma_init), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         self.conv02 = ScaledLayer(nn.Conv2d(width, width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale*sigma_init, base_width=base_width)
        
#         self.block2 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first, base_width=base_width, sigma_init=sigma_init), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         #self.conv03 = ScaledLayer(nn.Conv2d(width, width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale*sigma_init, base_width=base_width)

#         self.block3 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first, base_width=base_width, sigma_init=sigma_init), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
#         final_size =  self.img_dim//(8*init_stride) 
#         self.final_width = int(final_size**2 * width)
#         self.gamma = math.sqrt(self.final_width) if gamma == "sqrt_width" else 1.0
#         self.gamma = self.gamma * self.gamma_zero
        
#         sigma_init = 0.0 if self.zero_init_readout else sigma_init
#         self.fc = ScaledLayer(nn.Linear(self.final_width, num_classes, bias=bias), sigma_init=sigma_init*depth_scale, depth_scale=1/depth_scale, gamma=self.gamma)
                
#     def forward(self, x):
#         x = self.conv01(x)
#         x = nn.functional.relu(x)
#         for layer in self.block1:
#             x = self.skip_scaling*x + self.beta * layer(x)
#         x = F.max_pool2d(x, 2, 2) 
#         x = self.conv02(x)
#         for layer in self.block2:
#             x =  self.skip_scaling*x + self.beta * layer(x)
#         x = nn.functional.relu(x)
#         x = F.max_pool2d(x, 2, 2)
#         #x = self.conv03(x)
#         for layer in self.block3:
#             x =  self.skip_scaling*x + self.beta * layer(x)
#         x = x.view(-1, self.final_width)
#         x = self.fc(x)
        
#         return x
    
#     def set_to_lazy(self):
#         for branch in self.block1:
#             branch.set_to_lazy()
#         for branch in self.block2:
#             branch.set_to_lazy()
#         for branch in self.block3:
#             branch.set_to_lazy()
            
            
#     def get_module_classes_to_log(self):
#         return (ScaledResidualBranch, ScaledLayer)
    
    
    
class ConvNet(nn.Module):
    def __init__(self, width, n_blocks, res_scaling=1, skip_scaling=1, beta=1, gamma_zero=1, num_classes=10, img_dim=32,
                 norm=None, layers_per_block=1, zero_init_readout=False, non_lin_first=True, gamma='sqrt_width', init_stride=1,
                 depth_scale_non_res_layers=False, sigma_last_layer_per_block=1, bias=None, base_width=1.0):
        super().__init__()

        self.tot_n_blocks = n_blocks
        self.n_blocks = n_blocks // 3
        self.res_scaling = res_scaling
        self.skip_scaling = skip_scaling
        self.beta = beta
        self.gamma_zero = gamma_zero
        self.img_dim = img_dim
        self.zero_init_readout = zero_init_readout
        self.depth_scale_non_res_layers = depth_scale_non_res_layers
        
        depth_scale = 1.0 if not self.depth_scale_non_res_layers else math.sqrt(self.tot_n_blocks)

        self.conv01 = ScaledLayer(nn.Conv2d(3, width, 3, stride=init_stride, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)
        
        self.block1 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
        self.conv02 = ScaledLayer(nn.Conv2d(width, width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)
        
        self.block2 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
        self.conv03 = ScaledLayer(nn.Conv2d(width, width, 3, 1, padding=1, bias=bias), depth_scale=1/depth_scale, sigma_init=depth_scale)

        self.block3 = nn.ModuleList([ScaledResidualBranch(ResidualBranch(BlockLayer, width, width, n_layers=layers_per_block, sigma_init_last=sigma_last_layer_per_block, norm=norm, non_lin_first=non_lin_first), res_scaling=self.res_scaling) for _ in range(self.n_blocks)])
        
        final_size =  self.img_dim//(8*init_stride) 
        self.final_width = int(final_size**2 * width)
        self.gamma = math.sqrt(self.final_width / base_width) if gamma == "sqrt_width" else 1.0
        self.gamma = self.gamma * self.gamma_zero
        
        sigma_init = 0.0 if self.zero_init_readout else 1.0
        self.fc = ScaledLayer(nn.Linear(self.final_width, num_classes, bias=bias), sigma_init=sigma_init*depth_scale, depth_scale=1/depth_scale, gamma=self.gamma)
                
    def forward(self, x):
        x = self.conv01(x)
        for layer in self.block1:
            x = self.skip_scaling*x + self.beta * layer(x)
        x = F.max_pool2d(x, 2, 2) 
        x = self.conv02(x)
        for layer in self.block2:
            x =  self.skip_scaling*x + self.beta * layer(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv03(x)
        for layer in self.block3:
            x =  self.skip_scaling*x + self.beta * layer(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.final_width)
        x = self.fc(x)
        
        return x
    
    def set_to_lazy(self):
        for branch in self.block1:
            branch.set_to_lazy()
        for branch in self.block2:
            branch.set_to_lazy()
        for branch in self.block3:
            branch.set_to_lazy()
            
            
    def get_module_classes_to_log(self):
        #(ScaledResidualBranch, ScaledLayer)
        return (ScaledResidualBranch,)
    
    
    
    
class SimpleConvNet(nn.Module):
    def __init__(self, width, gamma_zero=1, num_classes=10, img_dim=32, zero_init_readout=False,  gamma='sqrt_width', bias=None, base_width=1.0, sigma_init=1):
        super().__init__()
        

        self.gamma_zero = gamma_zero
        self.img_dim = img_dim
        self.zero_init_readout = zero_init_readout

        
        self.conv01 = ScaledLayer(nn.Conv2d(3, width, 3, stride=2, padding=1, bias=bias), sigma_init=sigma_init)
        self.act1 = nn.ReLU()
        self.conv02 = ScaledLayer(nn.Conv2d(width, width, 3, padding=1, bias=bias, stride=4), sigma_init=sigma_init)
        self.act2 = nn.ReLU()
        
        final_size =  self.img_dim//32
        self.final_width = int(final_size**2 * width)
        self.gamma = math.sqrt(self.final_width/base_width) if gamma == "sqrt_width" else 1.0
        self.gamma = self.gamma * self.gamma_zero
        
        sigma_init = 0.0 if self.zero_init_readout else 1.0
        self.fc = ScaledLayer(nn.Linear(self.final_width, num_classes, bias=bias), sigma_init=sigma_init, gamma=self.gamma)
                
    def forward(self, x):
        x = self.conv01(x)
        x = self.act1(x)
        x = F.max_pool2d(x, 2, 2) 
        
        x = self.conv02(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, self.final_width)
        x = self.fc(x)
        
        return x
    
    def set_to_lazy(self):
        pass
            
            
    def get_module_classes_to_log(self):
        #(ScaledResidualBranch, ScaledLayer)
        return ()

