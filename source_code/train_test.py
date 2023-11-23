import wandb
import torch
import sys
from metrics import gradient_norm
from metrics import activation_norm_dict, entropies_dict

# Training
def train(epoch, batches_seen, nets, metrics, num_classes, trainloader, optimizers, criterion, device, schedulers, log=True, max_updates=-1, activations=None, get_entropies=False):
    #from utils import progress_bar
    print('\nEpoch: %d' % epoch)
    for e, net in enumerate(nets):
        net.train()

    E = len(nets)

    # if use_cut_mix_mixup:
    #     cutmix = v2.CutMix(num_classes=num_classes)
    #     mixup = v2.MixUp(num_classes=num_classes)
    #     cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        
    compute_every = 200
    train_loss = 0
    ens_train_loss = 0
    correct = 0
    total = 0
    correct_ens = 0
    total_ens = 0
    batches_seen = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # if use_cut_mix_mixup:
        #     inputs, targets = cutmix_or_mixup(inputs, targets)
        mean_logit = torch.zeros((targets.shape[0],num_classes)).to(device)
        for e, net in enumerate(nets):
            optimizers[e].zero_grad()
            outputs = net(inputs)
            #print(outputs.shape)
            #print(targets.shape)
            mean_logit += 1.0/E * outputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizers[e].step()
            train_loss += loss.item() / len(nets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        ens_train_loss += criterion(mean_logit, targets).item()
        total_ens += targets.size(0)
        _,predict_ens = mean_logit.max(1)
        correct_ens += predict_ens.eq(targets).sum().item()
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Ens Loss: %.3f | Acc: %.3f%% (%d/%d) | Ens Acc: %.3f%% (%d/%d)'
        #            % (train_loss/(batch_idx+1), ens_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_ens/total_ens, correct_ens, total_ens))
        #sys.stdout.write(f'\r batch = {batch_idx} , train loss = {train_loss/(batch_idx+1)}')
        
        if batches_seen % compute_every == 0 and batches_seen > 0:
            print("train_loss: {}, {}".format(train_loss, len(targets)))
            metrics['train_loss'] += [train_loss/compute_every]
            metrics['ens_train_loss'] += [ens_train_loss/compute_every]
            if total > 0 and total_ens > 0:
                metrics['train_acc'] += [100.0 * correct/total]
                metrics['ens_train_acc'] += [100.0 * correct_ens/total_ens] 
            
            if log:
                d_log = {
                    'train_loss': train_loss/compute_every,
                    'train_acc': 100.*correct/total,
                    'lr': schedulers[0].get_last_lr()[0] if len(schedulers)>0 else optimizers[0].param_groups[0]['lr']
                    }
                d_log.update(gradient_norm(net))
                if activations is not None:
                    d_log.update(activation_norm_dict(activations))
                    if get_entropies:
                        d_log.update(entropies_dict(activations))
                    d_log.update(net.relative_branch_norms(activations))
                wandb.log(d_log)
            train_loss = 0
            ens_train_loss = 0
            correct = 0
            total = 0
            correct_ens = 0
            total_ens = 0

            #for scheduler in schedulers:
            #    scheduler.step()
            #metrics = test(batches_seen, nets, metrics) # get test metrics
            #for e, net in enumerate(nets):
            #    net.train()
        if batches_seen >= max_updates and max_updates != -1:
            return metrics, batches_seen
            
        batches_seen += 1
        
        if len(schedulers) > 0:
            for scheduler in schedulers:
                scheduler.step()
        
    return metrics, batches_seen


def test(nets, metrics, num_classes, testloader, criterion, device):
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
    #all_logits_correct = [ [] for e in range(len(nets))]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mean_logit = torch.zeros((targets.shape[0],num_classes)).to(device)
            for e, net in enumerate(nets):
                outputs = net(inputs)
                # get the logit 
                #correct_logit = outputs[:,targets]
                #all_logits_correct[e] += [ correct_logit ] 
                #mean_logit += 1/len(nets) * outputs
                loss = criterion(outputs, targets)
                test_loss += loss.item()/len(nets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            ens_test_loss += criterion(mean_logit, targets).item()
            total_ens += targets.size(0)
            _,predict_ens = mean_logit.max(1)
            correct_ens += predict_ens.eq(targets).sum().item()
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Ens Loss: %.3f | Acc: %.3f%% (%d/%d) | Ens Acc: %.3f%% (%d/%d)'
            #         % (test_loss/(batch_idx+1), ens_test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_ens/total_ens, correct_ens, total_ens))
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Ens Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1),ens_test_loss/(batch_idx+1),100.*correct/total, correct, total, 100.*correct_ens/total_ens, correct_ens, total_ens))
    metrics['test_loss'] += [test_loss/(batch_idx+1)]
    metrics['ens_test_loss'] += [ens_test_loss/(batch_idx+1)]
    metrics['test_acc'] += [100.*correct/total]
    metrics['ens_test_acc'] += [100.*correct_ens/total_ens]
    
    return metrics
