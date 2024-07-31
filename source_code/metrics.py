import torch
import re
from torch.func import functional_call, vmap, jacrev
from pyhessian import hessian
import numpy as np
from asdl.kernel import kernel_eigenvalues
from torch.nn.functional import one_hot

def get_metrics_dict(hessian=True, hessian_rf=True, top_eig_ggn=False, top_k_dir_sharp=False, top_hessian_eigvals=1, ntk_eigs=0):
    metrics = []
    metrics_dict = {}
    c = False
    if hessian:
        metrics.extend(['trace'])
        for i in range(top_hessian_eigvals):
            metrics.extend([f"top_eig_{i}"]) 
        c = True
    if ntk_eigs > 0:
        for i in range(ntk_eigs):
            metrics.extend([f"ntk_eig_{i}"])
        c = True
    
    if hessian_rf:
        metrics.extend(["trace_rf", "top_eig_rf"])
        c = True  
    if top_eig_ggn:
        metrics.extend(["top_eig_ggn", "residual"])
        c = True
    if top_k_dir_sharp:
        # ks = [1, 2, 4, 6, 8, 10]
        # for k in ks:
            # metrics.extend([f"top_{k}_dir_sharp"])
        metrics.extend([f"directional_sharpness"])
        c = True
    metrics_loss = ['train_loss', 'ens_train_loss', 'test_loss', 'ens_test_loss']
    metrics_acc = ['test_acc', 'ens_test_acc', 'train_acc', 'ens_train_acc']

    for k in metrics:
        metrics_dict[k] = []
    for k in metrics_loss + metrics_acc:
        metrics_dict[k] = [np.nan] if c else []
    return metrics_dict
    
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


def activ_skewness_dict(activations):
    d = {}
    for name, activ in activations.items():
        norm = activ_skewness(activ)
        d["activ_skewness_" + name] = norm
    return d

def activ_skewness(activation):
    # batch_size, channels, width, height
    flat_act = torch.abs(activation.transpose(1,2).transpose(2,3).flatten(0,2)) # batch_size x width x height, n_channels
    maxs, _ = flat_act.max(1)
    medians, _ = flat_act.median(1)
    return torch.mean(maxs/medians)
        
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


# From: https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html

def empirical_ntk_jacobian_contraction(model, fnet_single, x1, x2):
    
    params = {k: v.detach() for k, v in model.named_parameters()}
    
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(model, params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(model, params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def fnet_single(model, params, x):
    return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)


def hessian_trace_and_top_eig(model, criterion, inputs, targets, top_n=1, cuda=True):
    hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=cuda)
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_n)
    trace = hessian_comp.trace()
    return top_eigenvalues, trace

def residual_and_top_eig_ggn(model, inputs, targets, use_mse_loss):
    inputs = inputs.cuda()
    targets = targets.cuda()
    top_eig_ggn = kernel_eigenvalues(model, inputs, cross_entropy=(not use_mse_loss), print_progress=False)[0].item()
    outputs = model(inputs)
    if not use_mse_loss:
        outputs = torch.softmax(outputs, dim=1)
    residual = (outputs - one_hot(targets)).norm(dim=1).mean(dim=0).item()
    return top_eig_ggn, residual
    

def ntk_features(fnet, params, x):
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x)
    jac1 = torch.concat([jac.flatten(2) for jac in jac1], dim=-1) # flatten to n_samples x n_classes x n_parameters per layer
    return jac1.mean(1) # mean over classes


def get_projected_gradients(gradient, eigenvectors):
    g_proj = torch.tensor([gradient@eigenv for eigenv in eigenvectors]).to("cuda")
    return g_proj
    
    
def process_eigenvectors(eigenvectors):
    
    es = []
    for e in eigenvectors:
        ne = torch.cat([v.flatten() for v in e], dim=0) # get each eigenvector of size n_parameters
        es.append(ne)
    return es

     
        
def top_k_hessian_alignment(projected_gradients, gradients, k):
    projected_gradients = projected_gradients[:k]
    dir_sharp = projected_gradients.norm()**2 / gradients.norm()**2
    
    return dir_sharp.item()


def top_k_dir_sharpness(projected_gradients, gradients, k, top_eigs):
    # normalize gradient    
    projected_gradients = projected_gradients[:k]
    g = projected_gradients/gradients.norm()
    
    eigvals_mat = torch.diag(torch.tensor(top_eigs[:k]).to("cuda"))
    dir_sharp = g@eigvals_mat@g
    
    return dir_sharp.item()


def process_gradients(grads):
    grads = [g.flatten() for g in grads]
    return torch.cat(grads, dim=0)

def directional_sharpness(gradients, model, inputs):
    # import pdb
    # pdb.set_trace()
    # gradients = [g/g.norm() for g in gradients]
    # gradients is num_layers tensors

    # now do hvp
    Hg = torch.autograd.functional.hvp(model, inputs, gradients)
    dir_sharpnes = gradients @ Hg

    return dir_sharpnes

def ntk_eigenvalues(model, inputs, targets, num_eigs): 
    inputs = inputs.cuda()
    targets = targets.cuda()
    # very important: cross_entropy always set to FALSE!
    ntk_eigenvalues = kernel_eigenvalues(model, inputs, cross_entropy=False, print_progress=False, top_n=num_eigs)
    return ntk_eigenvalues

def hessian_trace_and_top_eig_rf(model, criterion, inputs, targets, cuda=True):
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    trace_rf, top_eigenvalues_rf = hessian_trace_and_top_eig(model, criterion, inputs, targets, cuda=cuda)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return trace_rf, top_eigenvalues_rf

            