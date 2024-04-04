import wandb
import torch
import sys
from metrics import gradient_norm, hessian_trace_and_top_eig, hessian_trace_and_top_eig_rf, residual_and_top_eig_ggn, ntk_eigenvalues
from metrics import activation_norm_dict, entropies_dict, empirical_ntk_jacobian_contraction, fnet_single, activ_skewness_dict
from pyhessian import hessian
import numpy as np
from asdl.kernel import kernel_eigenvalues


# Training
def train(epoch, batches_seen, nets, metrics, num_classes, trainloader, optimizers, criterion, device, schedulers, log=True, max_updates=-1, activations=None, get_entropies=False, logging_steps=200, use_mse_loss=False,
          eval_inputs=None, eval_targets=None, eval_hessian_random_features=False, eval_hessian=False, top_eig_ggn=False, ntk_eigs=0):
    
    print('\nEpoch: %d' % epoch)
    for e, net in enumerate(nets):
        net.train()

    E = len(nets)
        
    compute_every = logging_steps
    train_loss = 0
    ens_train_loss = 0
    correct = 0
    total = 0
    correct_ens = 0
    total_ens = 0
    batches_seen = 0
    batches_to_log = 1
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # if use_mse_loss:
        #     targets = targets.float() / 10
        #     mean_logit = torch.zeros((targets.shape[0],)).to(device)
        # else:
        mean_logit = torch.zeros((targets.shape[0],num_classes)).to(device)
        for e, net in enumerate(nets):
            if use_mse_loss:
                targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
            optimizers[e].zero_grad()
            outputs = net(inputs)
            
            # if use_mse_loss:
            #     outputs = outputs.flatten()
                
            mean_logit += 1.0/E * outputs
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                raise ValueError("Loss is nan, quitting training")
            
            loss.backward()
            optimizers[e].step()
            train_loss += (loss.item() / len(nets))
            # if not use_mse_loss:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if use_mse_loss:
                _, t_max = targets.max(1)
            else:
                t_max = targets
            correct += predicted.eq(t_max).sum().item()
        ens_train_loss += criterion(mean_logit, targets).item()
        # if not use_mse_loss:
        total_ens += targets.size(0)
        _,predict_ens = mean_logit.max(1)
        correct_ens += predict_ens.eq(t_max).sum().item()
        
        if batches_seen % compute_every == 0: #and batches_seen > 0:
            print("train_loss: {}, {}".format(train_loss/(compute_every), len(targets)))
            metrics['train_loss'] += [train_loss/compute_every]
            metrics['ens_train_loss'] += [ens_train_loss/compute_every]

            optimizers[0].zero_grad()
            nets[0].eval()
            if eval_hessian_random_features:
                top_eigenvalues, trace = hessian_trace_and_top_eig_rf(nets[0], criterion, eval_inputs, eval_targets, cuda=True)
                metrics["trace_rf"] += [np.mean(trace)]
                metrics["top_eig_rf"] += [top_eigenvalues[-1]]
            if eval_hessian:
                top_eigenvalues, trace = hessian_trace_and_top_eig(nets[0], criterion, eval_inputs, eval_targets, cuda=True)
                metrics["trace"] += [np.mean(trace)]
                metrics["top_eig"] += [top_eigenvalues[-1]]
            if top_eig_ggn:
                top_eig_ggn, residual = residual_and_top_eig_ggn(nets[0], eval_inputs, eval_targets, use_mse_loss)
                metrics['residual'] += [residual]
                metrics['top_eig_ggn'] += [top_eig_ggn]
            if ntk_eigs > 0:
                top_ntk_eigs = ntk_eigenvalues(nets[0], eval_inputs, eval_targets, ntk_eigs)
                for i in range(ntk_eigs):
                    metrics[f"ntk_eig_{i}"] += [top_ntk_eigs[i].item()] 
            #metrics["ntk_trace"] += [empirical_ntk_jacobian_contraction(nets[0], fnet_single, eval_inputs, eval_targets)]
            
            # if not use_mse_loss:
            if total > 0 and total_ens > 0:
                metrics['train_acc'] += [100.0 * correct/total]
                metrics['ens_train_acc'] += [100.0 * correct_ens/total_ens] 
            
            if log:
                # if not use_mse_loss:
                d_log = {
                    'train_loss': train_loss/compute_every,
                    'train_acc': 100.*correct/total,
                    'lr': schedulers[0].get_last_lr()[0] if len(schedulers)>0 else optimizers[0].param_groups[0]['lr']
                    }
                # else:
                #     d_log = {
                #     'train_loss': train_loss/compute_every,
                #     'lr': schedulers[0].get_last_lr()[0] if len(schedulers)>0 else optimizers[0].param_groups[0]['lr']
                #     }
                #d_log.update(gradient_norm(net))
                if activations is not None:
                    d_log.update(activ_skewness_dict(activations))
                    # d_log.update(activation_norm_dict(activations))
                    # if get_entropies:
                    #     d_log.update(entropies_dict(activations))
                    # d_log.update(net.relative_branch_norms(activations))
                wandb.log(d_log)
            
            nets[0].train()
            
            train_loss = 0
            ens_train_loss = 0
            correct = 0
            total = 0
            correct_ens = 0
            total_ens = 0
            batches_to_log = 1

            #for scheduler in schedulers:
            #    scheduler.step()
            #metrics = test(batches_seen, nets, metrics) # get test metrics
            #for e, net in enumerate(nets):
            #    net.train()
        if batches_seen >= max_updates and max_updates != -1:
            return metrics, batches_seen
            
        batches_seen += 1
        batches_to_log += 1
        
        if len(schedulers) > 0:
            for scheduler in schedulers:
                scheduler.step()
        
    return metrics, batches_seen


def test(nets, metrics, num_classes, testloader, criterion, device, use_mse_loss):
    #from utils import progress_bar
    global best_acc
    for e,net in enumerate(nets):
        net.eval()
    test_loss = 0
    ens_test_loss = 0
    correct = 0
    total = 0
    correct_ens = 0
    total_ens = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mean_logit = torch.zeros((targets.shape[0],num_classes)).to(device)
            for e, net in enumerate(nets):
                if use_mse_loss:
                    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    raise ValueError("Loss is nan, quitting training")
                    exit(1)
                test_loss += loss.item()/len(nets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if use_mse_loss:
                    _, t_max = targets.max(1)
                else:
                    t_max = targets
                correct += predicted.eq(t_max).sum().item()
            ens_test_loss += criterion(mean_logit, targets).item()
            total_ens += targets.size(0)
            _,predict_ens = mean_logit.max(1)
            correct_ens += predict_ens.eq(t_max).sum().item()
    metrics['test_loss'] += [test_loss/(batch_idx+1)]
    metrics['ens_test_loss'] += [ens_test_loss/(batch_idx+1)]
    metrics['test_acc'] += [100.*correct/total]
    metrics['ens_test_acc'] += [100.*correct_ens/total_ens]
    
    return metrics



def eval(nets, num_classes, loader, criterion, device, use_mse_loss):
    #from utils import progress_bar
    global best_acc
    for e,net in enumerate(nets):
        net.eval()
    tot_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            for e, net in enumerate(nets):
                if use_mse_loss:
                    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    raise ValueError("Loss is nan, quitting training")
                    exit(1)
                tot_loss += loss.item()/len(nets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if use_mse_loss:
                    _, t_max = targets.max(1)
                else:
                    t_max = targets
                correct += predicted.eq(t_max).sum().item()
    
    return tot_loss/(batch_idx+1), 100.*correct/total
