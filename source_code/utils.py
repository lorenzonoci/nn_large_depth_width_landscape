'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision.transforms as transforms
import torchvision
from architectures.cifar_arch import ConvNet
#import architectures.resnet as resnet
from optim_utils import MuSGD
import torch.optim as optim
import numpy as np
from architectures.vit import ViT


def get_width(model_name, width_mult):
    if model_name == "conv":
        width = int(16 * width_mult)
    elif model_name == 'vit':
        width = int(64 * width_mult)
    else:
        raise ValueError()
    return width

def get_depth(model_name, depth_mult):
    if model_name == "conv":
        depth = int(3 * depth_mult)
    elif model_name == 'vit':
        depth = int(2 * 3 * depth_mult) # 2 --> one attention block, one MLP block, 3 --> base number of transformers blocks
    else:
        raise ValueError()
    return depth
                    
def process_args(args):
    # if args.arch == "conv":
    #     args.depth = int(3 * args.depth_mult)
    #     args.width = int(16 * args.width_mult)
    # elif args.arch == "conv_inv":
    #     args.depth = int(3 * args.depth_mult)
    #     args.width = int(16 * args.width_mult)
    # elif args.arch == "resnet" or args.arch=="resnet18":
    #     args.depth =  int(4 * args.depth_mult * 2)
    # elif args.arch == "vit":
    #     args.depth = int(2*3*args.depth_mult) # 2 --> one attention block, one MLP block, 3 --> base number of transformers blocks
    # else:
    #     raise ValueError()
    args.width = get_width(args.arch, args.width_mult)
    args.depth = get_depth(args.arch, args.depth_mult)
    
    if args.res_scaling_type == 'none':
        args.res_scaling = 1.0
    elif args.res_scaling_type == 'sqrt_depth':
        args.res_scaling = 1/np.sqrt(args.depth)
    elif args.res_scaling_type == 'depth':
        args.res_scaling = 1/args.depth
    else:
        raise ValueError("Invalid value for arg res_scaling: {}".format(args.res_scaling_type))

    if args.norm == 'none':
        args.norm = None
    elif args.norm not in ["ln", "bn", None]:
        raise ValueError("Wrong value for normalization layer {}".format(args.norm))
    
    return args


def get_model(arch, width, depth, args):
    
    if arch == "conv" and args.dataset == "imgnet":
        net = ConvNet(width=width, n_blocks=depth, gamma=args.gamma, 
                      res_scaling=args.res_scaling, skip_scaling=args.skip_scaling,
                      beta=args.beta, gamma_zero=args.gamma_zero, num_classes = 1000, img_dim = 224, norm=args.norm,
                      non_lin_first=True, layers_per_block=args.layers_per_block, sigma_last_layer_per_block=args.sigma_last_layer_per_block,
                      init_stride=2, depth_scale_non_res_layers=args.depth_scale_non_res_layers)    
    elif arch == "conv" and args.dataset == "cifar10":
        net = ConvNet(width=width, n_blocks=depth,  gamma=args.gamma, 
                      res_scaling=args.res_scaling, skip_scaling=args.skip_scaling,
                      beta=args.beta, gamma_zero=args.gamma_zero, norm=args.norm, layers_per_block=args.layers_per_block,
                      non_lin_first=True, sigma_last_layer_per_block=args.sigma_last_layer_per_block, init_stride=2,
                      depth_scale_non_res_layers=args.depth_scale_non_res_layers)
        
    # elif arch == "resnet" and args.dataset == "cifar10":
    #     net = resnet.Resnet10(num_classes=10, feat_scale=1, wm=width_mult, depth_mult=depth_mult, gamma=args.gamma, 
    #                           res_scaling=args.res_scaling, depth_scale_first=args.depth_scale_first, norm=args.norm)

    # elif arch == "resnet" and args.dataset == "imgnet":
    #     net = resnet.Resnet10(num_classes=1000, feat_scale=7**2, wm=width_mult, depth_mult=depth_mult, gamma=args.gamma,
    #                           res_scaling=args.res_scaling, depth_scale_first=args.depth_scale_first, norm=args.norm)


    # elif arch == "resnet_pool" and args.dataset == "imgnet":
    #     net = resnet.Resnet10_pool(num_classes=1000, feat_scale=7**2/4, wm=width_mult, depth_mult=depth_mult, gamma=args.gamma,
    #                           res_scaling=args.res_scaling, depth_scale_first=args.depth_scale_first, norm=args.norm)
    
    elif arch == "vit" and args.dataset == "cifar10":
        net = ViT(num_classes=10, image_size=32, patch_size=4, heads=8, width=width, depth=depth, gamma=args.gamma, 
                res_scaling=args.res_scaling, norm=args.norm, depth_scale_non_res_layers=args.depth_scale_non_res_layers, use_relu_attn=args.use_relu_attn)
        
    # elif arch == "resnet" and args.dataset=="tiny_imgnet":
    #     net = resnet.Resnet10(num_classes=200, feat_scale=1, wm=width_mult, depth_mult=depth_mult, gamma=args.gamma,
    #                           res_scaling=args.res_scaling, depth_scale_first=args.depth_scale_first, norm=args.norm, stride_first=2)
        
    # elif arch == "vit" and args.dataset == "tiny_imgnet":
    #     net = ViT(num_classes=200, image_size=64, patch_size=8, heads=8, wm=width_mult, depth_mult=depth_mult, gamma=args.gamma, 
    #             res_scaling=args.res_scaling, norm=args.norm)
    else:
        raise ValueError()
    return net


def get_lr(net, args):
    lr = args.lr * args.gamma_zero**2 * net.gamma**2
    if args.depth_scale_lr == "one_sqrt_depth":
        lr = lr / np.sqrt(args.depth)
    elif args.depth_scale_lr == "depth":
        lr = lr * args.depth
    return lr
    
    
def get_optimizers(nets, args):
    
    if args.optimizer == 'musgd':

        optimizers = [ optim.SGD(net.parameters(), lr=get_lr(net, args),
                            momentum=args.momentum,
                            weight_decay=args.weight_decay) for net in nets ]
        
        # optimizers = [ MuSGD(net.parameters(), lr=args.lr,
        #                     momentum=args.momentum,
        #                     weight_decay=args.weight_decay) for net in nets ]
    # elif args.optimizer == 'muadam':
    #     optimizers = [ MuAdam(net.parameters(), lr=args.lr) for net in nets ]
        
    elif args.optimizer == 'sgd':
        optimizers = [optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for net in nets]
    elif args.optimizer == 'muadam':
        optimizers = [optim.Adam(net.parameters(), lr=get_lr(net, args)) for net in nets]
    else:
        raise ValueError()
    return optimizers
    
def load_data(args, generator, seed_worker):
    if args.dataset == "imgnet":
        transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
        trainset = torchvision.datasets.ImageNet(
            root=args.data_path, split = 'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator, worker_init_fn = seed_worker)

        testset = torchvision.datasets.ImageNet(
            root=args.data_path, split='val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, generator=generator, worker_init_fn = seed_worker)
    
    elif args.dataset == "cifar10":
        if not  args.no_data_augm:
            print("Using data augmentation")
            transform_train = transforms.Compose([
                    transforms.Resize((32,32)), 
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(), 
                    transforms.RandomRotation(10),   
                    # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), 
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            print("NOT using data augmentation")
            transform_train = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_test = transforms.Compose([transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, transform=transform_train, download=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator, worker_init_fn = seed_worker)

        testset = torchvision.datasets.CIFAR10( 
            root=args.data_path, train=False, transform=transform_test, download=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, generator=generator, worker_init_fn = seed_worker)
        
              
    elif args.dataset == "tiny_imgnet":
        
        transform_mean = np.array([ 0.485, 0.456, 0.406 ])
        transform_std = np.array([ 0.229, 0.224, 0.225 ])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])
        
        trainset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator, worker_init_fn = seed_worker)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, generator=generator, worker_init_fn = seed_worker)
        
    return trainloader, testloader