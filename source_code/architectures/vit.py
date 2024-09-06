# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from torch.nn.init import constant_

# from mup import MuReadout
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from architectures.utils import MyIdentity
from architectures.utils import ScaledLayer, ScaledResidualBranch
import math 
import re 
from metrics import tensor_norm

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# def init_method_normal(sigma):
#     """Init method based on N(0, sigma)."""
#     def init_(tensor):
#         return nn.init.normal_(tensor, mean=0.0, std=sigma)
#     return init_

# classes

def get_norm_layer(width, norm):
    if norm == None:
        return MyIdentity()
    elif norm == "ln":
        return nn.LayerNorm(width)
    else:
        raise ValueError()


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = get_norm_layer(dim, norm)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            ScaledLayer(nn.Linear(dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            ScaledLayer(nn.Linear(hidden_dim, dim)),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, standparam=False, use_relu=False, shaped_attention=False, learnable_strength_pars=False):
        super().__init__()

        self.num_attention_heads = heads
        self.attention_head_size = int(dim / heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.shaped_attention = shaped_attention
        self.hidden_size = dim
        self.use_relu = use_relu
        self.num_patches = num_patches
        self.scale_attn_weights = 1.0 if not use_relu else 1/num_patches
        
        if use_relu:
            self.attend = nn.ReLU()
        else:
            self.attend = nn.Softmax(dim=-1)
            
        if standparam:
            self.scale = float(self.attention_head_size) ** -0.5
        else:
            self.scale = float(self.attention_head_size) ** -1
            
        if self.shaped_attention:
            self.gamma_1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=learnable_strength_pars)
            self.gamma_2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=learnable_strength_pars)
            self.gamma_3 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=learnable_strength_pars)

        self.query = ScaledLayer(nn.Linear(dim, self.all_head_size, bias=False))
        self.key = ScaledLayer(nn.Linear(dim, self.all_head_size, bias=False))


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        attention_probs = self.scale_attn_weights * self.attend(attention_scores)
        
        if self.shaped_attention:
            nonzero_tokens = (attention_mask == 0).view(attention_mask.shape[0], -1).float()
            nnz = nonzero_tokens.sum(-1).unsqueeze(1).unsqueeze(1)
            center_mat = torch.matmul(nonzero_tokens.unsqueeze(2), nonzero_tokens.unsqueeze(1))
            center_mat = center_mat * 1/nnz
            center_mat = center_mat.unsqueeze(1).expand(attention_probs.size())
            m = torch.diag_embed(nonzero_tokens)
            m = m.unsqueeze(1).expand(attention_probs.size())
            attention_probs_stable = self.gamma_1 * m + self.gamma_2 * attention_probs - self.gamma_3 * center_mat

        if self.shaped_attention:
            attention_probs = attention_probs_stable

        return attention_probs
    
    
class MySelfAttention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, standparam=False, use_relu=False, shaped_attention=False, learnable_strength_pars=False):
        super().__init__()

        self.num_attention_heads = heads
        self.attention_head_size = int(dim / heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.attention_weights = Attention(dim, num_patches, heads=heads, standparam=standparam, use_relu=use_relu, shaped_attention=shaped_attention, learnable_strength_pars=learnable_strength_pars)

        self.value = ScaledLayer(nn.Linear(dim, self.all_head_size, bias=False))


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_probs = self.attention_weights(hidden_states, attention_mask)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer
    
    
# class Attention(nn.Module):
#     def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., standparam=False, use_relu=False):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         #project_out = not (heads == 1 and dim_head == dim)

#         self.dim = dim
#         self.heads = heads
#         self.scale_attn_weights = 1.0 if not use_relu else math.sqrt(1/num_patches)
        
#         if standparam:
#             self.scale = float(dim_head) ** -0.5
#         else:
#             self.scale = float(dim_head) ** -1
            
#         if use_relu:
#             self.scale=1

#         if use_relu:
#             self.attend = nn.ReLU()
#         else:
#             self.attend = nn.Softmax(dim=-1)
            
#         self.to_qkv = ScaledLayer(nn.Linear(dim, inner_dim*3, bias=False))

#         self.to_out = nn.Sequential(
#             ScaledLayer(nn.Linear(inner_dim, dim)),
#             nn.Dropout(dropout)
#         )
        
#         if not use_relu:
#             self._reset_parameters()
        
        
#     def _reset_parameters(self):
#         # zero initializing query head
#         constant_(self.to_qkv.layer.weight[:self.dim], 0.)
#         if self.to_qkv.layer.bias is not None:
#             constant_(self.to_qkv.layer.bias, 0.)
            
            
#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.scale_attn_weights * self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, num_patches, dropout = 0., res_scaling=1.0, norm='ln', use_relu_attn=False):
        super().__init__()
        self.res_scaling = res_scaling
        self.attn = ScaledResidualBranch(PreNorm(dim, MySelfAttention(dim, num_patches=num_patches, heads=heads, use_relu=use_relu_attn), norm), res_scaling=res_scaling)
        self.ffn = ScaledResidualBranch(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), norm), res_scaling=res_scaling)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, num_patches, dropout = 0., res_scaling=1.0, norm='ln', use_relu_attn=False):
        super().__init__()
        self.res_scaling = res_scaling
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim, num_patches, dropout, res_scaling, norm, use_relu_attn) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Embedding(nn.Module):
    def __init__(self, image_size, patch_size=4, width=64, channels = 3, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        dim = width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x
        
class ViT(nn.Module):
    def __init__(self, *, image_size, num_classes, patch_size=4, width=64, depth=3, heads=6, mlp_dim=128,
                 pool = 'cls', channels = 3, dim_head = -1, dropout = 0., emb_dropout = 0., 
                 gamma='sqrt_width', res_scaling=1, norm='ln', zero_init_readout=False, depth_scale_non_res_layers=False, use_relu_attn=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.depth_scale_non_res_layers = depth_scale_non_res_layers

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        dim = width
        mlp_dim = mlp_dim
        depth = depth // 2 # Depth provided is number of res layers, each block has 2 res layers (one attention, one mlp)
        
        if dim_head == -1:
            dim_head = dim // heads
        
        self.embedding = Embedding(image_size, patch_size, width, channels, emb_dropout)
        
        self.zero_init_readout = zero_init_readout
        
        self.transformer = Transformer(dim, depth, heads, mlp_dim, num_patches, dropout=dropout, res_scaling=res_scaling, norm=norm, use_relu_attn=use_relu_attn)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.gamma = math.sqrt(dim) if gamma == "sqrt_width" else 1.0
        depth_scale = 1.0 if not self.depth_scale_non_res_layers else math.sqrt(self.tot_n_blocks)
        sigma_init = 0.0 if self.zero_init_readout else 1.0
        
        self.mlp_head = nn.Sequential(
            get_norm_layer(dim, norm),
            ScaledLayer(nn.Linear(dim, num_classes), sigma_init=sigma_init*depth_scale, depth_scale=1/depth_scale, gamma=self.gamma)
        )

    # def init_weights(self):
    #     if self.mlp_head[1].bias is not None:
    #         self.mlp_head[1].bias.data.zero_()
    #     self.mlp_head[1].weight.data.zero_()
            
    def forward(self, img):
        x = self.embedding(img)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
    def get_module_classes_to_log(self):
        return (Embedding, TransformerBlock, ScaledResidualBranch, ScaledLayer, Attention)
    
    
    def relative_branch_norms(self, activations):
        norms = {}
        for name, activ in activations.items():
            match = re.search(r'transformer\.layers\.\d+\.attn$', name) #see if it is attention layer
            if match:
                layer_index = int(re.findall(r'\d+', name)[0]) #get layer index
                if layer_index == 0: # is it's first layer, the input is the embedding
                    input_to_attn = "embedding"
                else:
                    input_to_attn = "transformer.layers.{}".format(layer_index - 1)
                norms["relative_branch_norm_{}".format(layer_index)] = tensor_norm(activ) / tensor_norm(activations[input_to_attn])
        return norms