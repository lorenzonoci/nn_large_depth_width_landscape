import pandas as pd
import torch 
from tqdm import tqdm
from utils import get_model, get_width, get_depth, get_optimizers
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils import process_args
from metrics import activations_norm_to_df, register_activation_hooks


def parametr_check_weight_space(model_name, width_mult, depth_mults, dataloader, device, criterion, args, save_folder, n_steps=1, n_seeds=5):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    dfs = []
    for s in tqdm(range(n_seeds)):
        torch.manual_seed(s)
        for i, depth_mult in enumerate(depth_mults):
            args.depth_mult = depth_mult
            args.width_mult = width_mult
            args = process_args(args)
            
            model = get_model(model_name, args.width, args.depth, args).to(device)
            
            optimizer = get_optimizers([model], args)[0]
            
            df = get_weight_update(model, dataloader, device, criterion, optimizer, n_steps)
            if i == 0:
                layer_names = df["layer"].unique()
            df["depth_mult"] = depth_mult
            df["width_mult"] = width_mult
            dfs.append(df[df["layer"].isin(layer_names)])
    df = pd.concat(dfs).reset_index(drop=True)

    plot_coord(df, n_steps, save_folder, is_width=False)
    
    

def parametr_check_pl(model_name, width_mult, depth_mults, dataloader, device, criterion, args, save_folder, n_steps=1, n_seeds=5):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    dfs = []
    for s in tqdm(range(n_seeds)):
        torch.manual_seed(s)
        for i, depth_mult in enumerate(depth_mults):
            args.depth_mult = depth_mult
            args.width_mult = width_mult
            args = process_args(args)
            
            model = get_model(model_name, args.width, args.depth, args).to(device)
            model_pl = get_model(model_name, args.width, args.depth, args).to(device)
            model_pl.load_state_dict(model.state_dict()) # Copying initialization of the other model
            model_pl.set_to_lazy() # Setting copied model to lazy in the first K-1 layers of each residual branch 
            
            optimizer = get_optimizers([model], args)[0]
            optimizer_pl = get_optimizers([model_pl], args)[0]
            
            df = get_magnitude_feature_diff(model, model_pl, dataloader, device, criterion, optimizer, optimizer_pl, n_steps)
            if i == 0:
                layer_names = df["layer"].unique()
            df["depth_mult"] = depth_mult
            df["width_mult"] = width_mult
            dfs.append(df[df["layer"].isin(layer_names)])
    df = pd.concat(dfs).reset_index(drop=True)

    plot_coord(df, n_steps, save_folder, is_width=False)
    

def parametr_check_depth(model_name, width_mult, depth_mults, dataloader, device, criterion, args, save_folder, n_steps=1, n_seeds=5):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    dfs = []
    for s in tqdm(range(n_seeds)):
        torch.manual_seed(s)
        for i, depth_mult in enumerate(depth_mults):
            args.depth_mult = depth_mult
            args.width_mult = width_mult
            args = process_args(args)
            model = get_model(model_name, args.width, args.depth, args).to(device)
            optimizer = get_optimizers([model], args)[0]
            df = get_magnitude_update(model, dataloader, device, criterion, optimizer, n_steps)
            if i == 0:
                layer_names = df["layer"].unique()
            df["depth_mult"] = depth_mult
            df["width_mult"] = width_mult
            dfs.append(df[df["layer"].isin(layer_names)])
    df = pd.concat(dfs).reset_index(drop=True)

    plot_coord(df, n_steps, save_folder, is_width=False)


def parametr_check_width(model_name, width_mults, depth_mult, dataloader, device, criterion, args, save_folder, n_steps=1, n_seeds=5):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    dfs = []
    for s in tqdm(range(n_seeds)):
        torch.manual_seed(s)
        for width_mult in width_mults:
            args.depth_mult = depth_mult
            args.width_mult = width_mult
            args = process_args(args)
            model = get_model(model_name, get_width(model_name, width_mult), get_depth(model_name, depth_mult), args).to(device)
            optimizer = get_optimizers([model], args)[0]
            df = get_magnitude_update(model, dataloader, device, criterion, optimizer, n_steps)
            df["depth_mult"] = depth_mult
            df["width_mult"] = width_mult
            dfs.append(df)        
    df = pd.concat(dfs).reset_index(drop=True)
    plot_coord(df, n_steps, save_folder, is_width=True)

    
def plot_coord(df, n_steps, save_folder, is_width=True):
    x = "width_mult" if is_width else "depth_mult"
    for t in range(n_steps):
        df_t = df[df["step"]==t]
        plt.figure(figsize = (10,6))
        sns.lineplot(df_t, x=x, y="norm", hue="layer",  marker='o', markersize=8, palette=sns.color_palette("rocket", n_colors=len(df["layer"].unique())))
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "coord_check_{}_step_{}.pdf".format(x, t)))
        plt.close()
        
        
        
def get_weight_update(model, dataloader, device, criterion, optimizer, n_steps=1):
    
    model.train()
    df = pd.DataFrame(columns=["step", "layer", "norm"])
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        break

    # initialization
    model1_params = get_parameter_dict(model)
    for t in range(n_steps):
        outputs = model(inputs) 
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        model2_params = get_parameter_dict(model)
        df = activations_norm_to_df(df, model1_params, model2_params, t)
    return df

 
def get_magnitude_update(model, dataloader, device, criterion, optimizer, n_steps=1):
    
    model.train()
    df = pd.DataFrame(columns=["step", "layer", "norm"])
    
    activations = register_activation_hooks(model)
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        break

    # initialization
    outputs = model(inputs) 
    activations_t1 = activations.copy()
    
    for t in range(n_steps):
        
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        outputs = model(inputs)
        
        activations_t2 = activations.copy()
        activations_norm_to_df(df, activations_t1, activations_t2, t)
        activations_t1 = activations_t2
        
    return df


def get_magnitude_feature_diff(model1, model2, dataloader, device, criterion, optimizer1, optimizer2, n_steps=1):
    
    model1.train()
    model2.train()
    
    df = pd.DataFrame(columns=["step", "layer", "norm"])
    
    activations1 = register_activation_hooks(model1)
    activations2 = register_activation_hooks(model2)
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        break

    # initialization
    #activations_t1 = activations.copy()
    outputs1 = model1(inputs) 
    outputs2 = model2(inputs)
    
    for t in range(n_steps):
        
        optimizer1.zero_grad()
        loss = criterion(outputs1, targets)
        loss.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss = criterion(outputs2, targets)
        loss.backward()
        optimizer2.step()
        
        outputs1 = model1(inputs) 
        outputs2 = model2(inputs)
        activations_norm_to_df(df, activations1, activations2, t)
    print(loss)
    return df
       
       
def get_parameter_dict(model):
    d = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "conv" not in name:
            d[name] = param.data.detach().cpu()
    return d