from collections import defaultdict

from torch.optim import SGD, Adam, AdamW
from torch.nn.init import _calculate_fan_in_and_fan_out

# Code adapted from mup package (Microsoft)

def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups


def get_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(1)
    fan_out = tensor.size(0)
    return fan_in, fan_out
    

def MuSGD(params, **kwargs):
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        def new_group():
            new_g = {k:v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g
        
        lrs_mult_dict = defaultdict(new_group) # key is width mult
        fixed_p = new_group()
        
        for i,p in enumerate(param_group['params']):
            dims = p.dim()
            if dims > 1:
                fan_in, fan_out = get_fan_in_and_fan_out(p)
                #r = fan_in / fan_out
                # TODO: adjust for different fan_in fan_out
                if i == 0:
                    lrs_mult_dict[fan_out]['params'].append(p) # rescale first layer by N (fan out)
                else:
                    lrs_mult_dict[fan_in]['params'].append(p)
            else:
                fixed_p['params'].append(p)
        for width, group in lrs_mult_dict.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] *= width
            group['weight_decay'] /= width
        new_param_groups.extend(list(lrs_mult_dict.values()) + [fixed_p])
    return SGD(new_param_groups, **kwargs)
