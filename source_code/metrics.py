import torch
import re

def activations_norm_to_df(df, activations_t1, activations_t2, step):
    for name, activ_t1 in activations_t1.items():
        activ_t2 = activations_t2[name]
        norm = tensor_norm(activ_t1 - activ_t2)
        df.loc[len(df.index), :] = [step, name, norm] 
    return df
  
def activation_norm_dict(activations):
    d = {}
    for name, activ in activations.items():
        norm = tensor_norm(activ)
        d["activ_" + name] = norm
    return d
        
def tensor_norm(activation):
    n_elem = activation.numel()
    norm = 1/n_elem * float(torch.norm(activation, p=2).square().cpu())
    return norm
       
        
def register_activation_hooks(model):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    for name, layer in model.named_modules():
        if layer_conditions(layer, model):
            layer.register_forward_hook(get_activation(name))
    return activations

def layer_conditions(layer, model):
    c = isinstance(layer,  model.get_module_classes_to_log())
    return c 


def gradient_norm(model):
    d = {}
    for name, param in model.named_parameters():
        if "bias" not in name:
            d["grad_" + name] = tensor_norm(param.grad.data)
    return d


def get_entropy(
    x,
    return_super_raw=False,
    return_raw=False,
    return_variance=True,
    already_softmaxed=False,
):
    # assume that shape of tensor is (batch_size, num_heads, sequence_length, sequence_length)
    if not already_softmaxed:
        x = x.softmax(dim=-1)

    # entropy per row
    x_entr = torch.sum(torch.special.entr(x), axis=-1)

    if return_super_raw:
        return x_entr

    x_entr = torch.mean(x_entr, axis=0)

    # now size is (num_heads, sequence_length)
    if return_raw:
        return x_entr

    entr_heads = torch.mean(x_entr, axis=-1)
    tot_mean_entr = torch.mean(entr_heads)

    if not return_variance:
        return tot_mean_entr, entr_heads

    # only considering the variance coming from token dimension and head (no batch)
    entr_head_var = torch.var(x_entr, axis=-1)
    tot_var_entr = torch.var(x_entr)
    return entr_heads, tot_mean_entr, entr_head_var, tot_var_entr


def entropies_dict(activations):
    entropies = {}
    for name, activ in activations.items():
        match = re.search(r'.*attention_weights$', name)
        if match:
            tot_mean_entr, _ = get_entropy(activ, already_softmaxed=True, return_variance=False)
            entropies["entr_" + name] = tot_mean_entr
    return entropies


# def sharpness():
            