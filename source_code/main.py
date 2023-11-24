import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
import wandb
from utils import load_data, get_model, get_optimizers, process_args
from train_test import train, test
from functools import partial
import transformers
from test_parametr import parametr_check_width, parametr_check_depth, parametr_check_pl, parametr_check_weight_space
from metrics import register_activation_hooks


wandb_project_name = 'optimize_depth_scale_arch'
wand_db_team_name = "large_depth_team"

def get_run_name(args):
    return "model_{}/optimizer{}/dataset_{}/epoch_{}/lr_{:.6f}/seed_{}/momentum_{}/batch_size_{}/res_scaling_{}/width_mult_{}/depth_mult_{}/skip_scaling_{}/beta_{}/gamma_zero_{}/weight_decay_{}/norm_{}/k_layers_{}".format(
        args.arch, args.optimizer, args.dataset, args.epochs, args.lr, args.seed, args.momentum, args.batch_size, args.res_scaling_type, args.width_mult, args.depth_mult,
        args.skip_scaling, args.beta, args.gamma_zero, args.weight_decay, args.norm, args.layers_per_block)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=''
    '''
    PyTorch implementation of various parametrizations for neural networks.
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--parametr', type=str, default='none', choices=['sp', 'mup', 'mup_sqrt_depth', 'mup_depth', 'none'],help=
                        '''
                        set all the parameters required for a specified parametrization. 
                        1. sp: standard parametrization
                        2. mup: mup parametrization (rescale last layer by 1/sqrt(width) and learning rate by width)
                        3. mup_sqrt_depth: in addition to mup, also scale residual branches by 1/sqrt{depth}
                        4. mup_depth: in addition to mup, scale residual branches by 1/depth, learning rate by depth and scale 
                                      the layers in non-residual blocks by 1/sqrt{depth} and scale std of their initialization by sqrt(depth)
                        5. none: choose this if you do not want to use any of these parametrizations and set all the flags manually.
                        
                        If you do not choose 'none', the flags that this command overrides are: 'res_scaling_type', 'depth_scale_lr', 
                            'depth_scale_non_res_layers', 'gamma'. In addition, 'optimizer' is adjusted to match the parametrization. 
                        ''')
    parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
    parser.add_argument('--arch', type=str, default='conv')
    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'adam', 'musgd', 'muadam'])
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--width_mult', type=float, default=2.0)
    parser.add_argument('--save_dir', type=str, default='test/',
                    help='file location to save results')
    parser.add_argument('--res_scaling_type', type=str, default='none')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='imgnet')
    parser.add_argument('--depth_mult', type=int, default=1)
    parser.add_argument('--skip_scaling', type=float, default=1,
                         help='set to zero to use an MLP without skip connections')
    parser.add_argument('--beta', type=float, default=1,
                         help='scaling factor for the residual branch. To use together with res_scaling parameter')
    parser.add_argument('--base_width', type=float, default=1, 
                        help='every 1/sqrt{N} factor is upscaled by constant equal to base_width')
    parser.add_argument('--gamma', type=str, default='none',
                         help='')
    parser.add_argument('--gamma_zero', type=float, default=1,
                         help='controls the amount of feature learning.')
    parser.add_argument('--norm', type=str, default='none',
                         help='normalization layer')
    parser.add_argument('--schedule', action='store_true',help ='cosine anneal schedule')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--depth_scale_lr', type=str, help ='scale learning rate by 1/sqrt(depth) or by depth', choices=['one_sqrt_depth', 'depth', 'none'])
    parser.add_argument('--layers_per_block', type=int, default=1)
    parser.add_argument('--depth_scale_non_res_layers', action='store_true', 
                        help='For the layers that are not : scale the std of the weights by sqrt(depth) and divide outside by sqrt(depth)')
    parser.add_argument('--sigma_last_layer_per_block', type=float, default=1,
                         help='standard deviation of the weights of the final layer of each block')
    parser.add_argument('--no_data_augm',  action='store_true',
                         help='use data aumentation')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--num_ens', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--test_num_workers', type=int, default=16)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--multiprocessing', action='store_true',
                         help='distributed computation for multiple GPU (single node)')
    parser.add_argument('--test_parametr', action='store_true')
    parser.add_argument('--use_relu_attn', action='store_true')
    parser.add_argument('--log_activations', action='store_true')
    parser.add_argument('--no_wandb', action='store_false')
    args = parser.parse_args()
    
    
    if args.parametr == 'sp':
        args.res_scaling_type = 'none'
        args.depth_scale_lr = 'none'
        args.depth_scale_non_res_layers = False
        args.optimizer = 'adam' if 'adam' in args.optimizer else 'sgd'
        args.gamma = 'none'
        
    elif args.parametr == 'mup':
        args.res_scaling_type = 'none'
        args.depth_scale_lr = 'one_sqrt_depth' if "adam" in args.optimizer else 'none'
        args.depth_scale_non_res_layers = False
        args.optimizer = 'muadam' if 'adam' in args.optimizer else 'musgd'
        args.gamma = 'sqrt_width'
        
    elif args.parametr == 'mup_sqrt_depth':
        args.res_scaling_type = 'sqrt_depth'
        args.depth_scale_lr = 'one_sqrt_depth' if "adam" in args.optimizer else 'none'
        args.depth_scale_non_res_layers = False
        args.optimizer = 'muadam' if 'adam' in args.optimizer else 'musgd'
        args.gamma = 'sqrt_width'
    
    elif args.parametr == 'mup_depth':
        args.res_scaling_type = 'depth'
        args.depth_scale_lr = 'none' if "adam" in args.optimizer else 'depth'
        args.depth_scale_non_res_layers = True
        args.optimizer = 'muadam' if 'adam' in args.optimizer else 'musgd'
        args.gamma = 'sqrt_width'
    
    elif args.parametr != 'none':
        raise ValueError()
    
    c = 0
    if args.lr == -1:
        lrs = np.logspace(-2.5, 1.5, num=19)[4:14] if "adam" not in args.optimizer else np.logspace(-4, -2, num=10)
        c += 1
    else:
        lrs = [args.lr]
        
    if args.batch_size == -1:
        batch_sizes = np.logspace(4, 9, num=6, base=2)
        c += 1
    else:
        batch_sizes = [args.batch_size]
        
    if args.momentum == -1:
        momenta = np.linspace(0.1, 1, num=10)
        c += 1
    else:
        momenta = [args.momentum]
        
    if args.width_mult == -1:
        width_mults = np.logspace(1, 4, num=4, base=2)
    else:
        width_mults = [args.width_mult]
        
    if args.beta == -1:
        betas = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    else:
        betas = [args.beta]
    
    if args.gamma_zero == -1:
        gamma_zeros = np.linspace(0.3, 1, num=8)
    else:
        gamma_zeros = [args.gamma_zero]

    if args.weight_decay == -1:
        lambdas = [1e-5,1e-4,1e-3,1e-2,1e-1]
    else:
        lambdas = [args.weight_decay]
    
    if c > 1:
        print(f"Warning: performing hyperparameter search over {c} parameters. It might take a while")
    
    max_updates = -1

    for gamma_zero in gamma_zeros:
        for beta in betas:
            for width_mult in width_mults:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for momentum in momenta:
                            for lamb in lambdas:
                                args.batch_size = int(batch_size)
                                args.momentum = momentum
                                args.width_mult = int(width_mult)
                                args.beta = beta
                                args.gamma_zero = gamma_zero
                                args.weight_decay = lamb
                                args.lr = lr

                                print(f"Proccesing hyperparmeters: learning rate {args.lr}, batch size {args.batch_size}, momentum {args.momentum}, \
                                      width_mult {args.width_mult}, beta {args.beta}, gamma_zero {args.gamma_zero} weight_decay {args.weight_decay}")
                                print(args)
                                ## TODO: CODE THIS BETTER
                                if args.dataset == "imgnet":
                                    num_classes = 1000 
                                elif args.dataset == "tiny_imgnet":
                                    num_classes = 200
                                elif args.dataset == "cifar10":
                                    num_classes = 10
                                else:
                                    raise ValueError()
                                
                                # logs
                                run_name = get_run_name(args)

                                if not os.path.isdir(args.save_dir):
                                    os.mkdir(args.save_dir)
                                args.save_path = os.path.join(args.save_dir, run_name.replace("/", "-"))
                                if not os.path.isdir(args.save_path):
                                    os.mkdir(args.save_path)

                                if not args.no_wandb:
                                    wandb.init(
                                    # set the wandb project where this run will be logged
                                    entity=wand_db_team_name,
                                    project=wandb_project_name,

                                    # track hyperparameters and run metadata
                                    config=args.__dict__
                                    )
                                    wandb.run.name = run_name

                                args = process_args(args)
                                
                                if len(lrs) == 1 and len(batch_sizes) > 1:
                                    print("Setting learning rate based on batch size")
                                    args.lr = args.lr * batch_size / batch_sizes[0]
                                    
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                print("device = " + device)
                                if args.multiprocessing:
                                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


                                best_acc = 0  # best test accuracy
                                start_epoch = 0  # start from epoch 0 or last checkpoint epoch

                                E = args.num_ens
                                # Set the random seed manually for reproducibility.
                                torch.manual_seed(args.seed)

                                # Data
                                print('==> Preparing data..')

                                g = torch.Generator()
                                g.manual_seed(args.seed)

                                def seed_worker(worker_id):
                                    worker_seed = torch.initial_seed() % 2**32
                                    np.random.seed(worker_seed)
                                    random.seed(worker_seed)
                                trainloader, testloader = load_data(args, generator=g, seed_worker=seed_worker)
                                if len(batch_sizes) > 1 and max_updates == -1:
                                    # epochs x n_batches
                                    max_updates = args.epochs * len(trainloader) # calculate n updates based on first batch size
                                    args.epochs = 1000 # anyway it will break before
                                    print(f"Training for {max_updates} steps")


                                # Model
                                print('==> Building model..')
                                nets = []
                                for e in range(E):
                                    torch.manual_seed(e)
                                    nets.append(get_model(args.arch, args.width, args.depth, args))
                                print(nets[0])
                                if args.log_activations:
                                    activations = register_activation_hooks(nets[0])
                                else:
                                    activations = None
                                if args.multiprocessing == True:
                                    # assumes GPUs on a single node
                                    device_ids = [i for i in range(torch.cuda.device_count())]
                                    print("DEVICE IDs")
                                    print(device_ids)
                                    for net in nets:
                                        net = torch.nn.DataParallel(net, device_ids)
                                
                                nets = [net.to(device) for net in nets]

                                torch.manual_seed(args.seed)
                                torch.cuda.manual_seed(args.seed)

                                if args.resume:
                                    # Load checkpoint.
                                    print('==> Resuming from checkpoint..')
                                    checkpoint = torch.load(os.path.join(args.save_path, f"model_ckpt_N_{args.width_mult}_last_.pth"))
                                    state = torch.load(os.path.join(args.save_path + f'/ckpt_N_{args.width_mult}_batches_{args.resume_epochs}_.pth'))
                                    nets_weights = checkpoint['nets']
                                    [net.load_state_dict(net_weights) for (net, net_weights) in zip(nets, nets_weights)]
                                    start_epoch = state['epoch'] + 1

                                criterion = nn.CrossEntropyLoss()
                                
                                if args.test_parametr:
                                    #parametr_check_weight_space(args.arch, 2, [1, 2, 4, 8, 16, 32, 64], trainloader, device, criterion, args, n_steps=10, save_folder="./coord_check_weight_{}".format(args.parametr))
                                    #parametr_check_pl(args.arch, 2, [1, 2, 4, 8, 16, 32, 64, 128], trainloader, device, criterion, args, n_steps=5, save_folder="./coord_check_{}_pl".format(args.parametr), n_seeds=10)
                                    parametr_check_depth(args.arch, 2, [1, 2, 4, 8, 16, 32], trainloader, device, criterion, args, n_steps=10, save_folder="./coord_check_{}".format(args.parametr))
                                    parametr_check_width(args.arch, [2, 4, 8, 16, 32], 1, trainloader, device, criterion, args, n_steps=10, save_folder="./coord_check_{}".format(args.parametr))
                                    #exit()
                                
                                optimizers = get_optimizers(nets, args)

                                if args.schedule:
                                    scheduler = partial(
                                        transformers.get_cosine_schedule_with_warmup,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.epochs * len(trainloader),
                                        num_cycles=0.5,
                                    )
                                    schedulers = [scheduler(optimizer) for optimizer in optimizers]
                                elif args.warmup_steps > 0:
                                    def wu_scheduler(opt, warmup_steps):
                                        return torch.optim.lr_scheduler.LambdaLR(
                                                opt,
                                                lr_lambda=lambda step: min(
                                                    1.0, step / warmup_steps
                                                ),  # Linear warmup over warmup_steps.
                                                )
                                    schedulers = [wu_scheduler(optimizer, args.warmup_steps) for optimizer in optimizers]
                                else:
                                    schedulers = []

                                metrics = {'train_loss': [], 'ens_train_loss': [], 'test_loss': [], 'ens_test_loss': [], 'test_acc': [], 'ens_test_acc': [], 'train_acc': [], 'ens_train_acc': []}
                                batches_seen = 0
                                for epoch in range(start_epoch, start_epoch+args.epochs):
                                    metrics, batches_seen = train(epoch,batches_seen,nets,metrics, num_classes, trainloader, optimizers, criterion, device, schedulers, log=True, max_updates=max_updates, activations=activations, get_entropies=True)
                                    metrics = test(nets, metrics, num_classes, testloader, criterion, device)
                                    
                                    print('Saving..')
                                    state = {
                                        'metrics': metrics,
                                        'epoch': epoch
                                    }
                                    if not os.path.isdir(args.save_path):
                                        os.mkdir(args.save_path)
                                    torch.save(state, args.save_path + f'/ckpt_N_{args.width_mult}_batches_{epoch}_.pth')    
                                    net_state = {'nets': [net.state_dict() for net in nets]}
                                    torch.save(net_state, args.save_path + f'/model_ckpt_N_{args.width_mult}_last_.pth')
                                    if batches_seen >= max_updates and max_updates!=-1:
                                        print("exiting")
                                        break
                                if not args.no_wandb:
                                    wandb.finish()
        

