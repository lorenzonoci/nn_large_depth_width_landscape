#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --optimizer muadam --save_dir=yy/home/ameterez/work/nips2024/understanding_adam_wider --dataset cifar10  --epochs 50  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100 --top_hessian_eigvals 1 --eval_hessian

# python main.py  --arch vit --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --save_dir=/home/ameterez/work/nips2024/vit_training_without_lr_scaling --dataset cifar10  --epochs 20 --batch_size 64 --logging_steps 100 --optimizer muadam --num_workers 16 --warmup_steps 500 --top_hessian_eigvals 1 --norm ln --eval_hessian


# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --optimizer muadam --save_dir=/home/ameterez/work/nips2024/understanding_multiple_layers_per_block_longer_with_adam_no_scaling --dataset cifar10  --epochs 50  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100 --top_hessian_eigvals 1 --eval_hessian

# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --save_dir=/home/ameterez/work/nips2024/ntk_and_hessian_width_clean --dataset cifar10  --epochs 25  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100 --num_workers 16 --top_hessian_eigvals 5 --ntk_eigs 5 --skip_scaling 0 --eval_hessian

# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5   --layers_per_block $6 --save_dir=/home/ameterez/work/nips2024/lorenzo_command_beta_2_k_3_full --dataset cifar10  --epochs 80  --beta 2 --batch_size 128 --logging_steps 100 --warmup_steps 100 --num_workers 16 --top_hessian_eigvals 1 --eval_hessian

# this is the experiment that we were talking about with long time training
# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --save_dir=/home/ameterez/work/nips2024/test_after_submission --dataset cifar10  --epochs 300  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 80 --warmup_steps 100 --num_workers 16 --top_hessian_eigvals 5 --ntk_eigs 5 --eval_hessian --skip_scaling 0


# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --optimizer muadam --save_dir=/fast/ameterez/nips2024_rebuttal/testing_adam_no_hessian --dataset cifar10  --epochs 30  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100


# python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --layers_per_block $6 --save_dir=/fast/ameterez/nips2024_rebuttal/directional_sharpness_convnets_width --dataset cifar10  --epochs 20 --beta 1 --batch_size 256 --logging_steps 80 --warmup_steps 100 --num_workers 16 --eval_dir_sharpness --skip_scaling 0

# python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --layers_per_block $6 --save_dir=/fast/ameterez/nips2024_rebuttal/directional_sharpness_convnets_depth --dataset cifar10  --epochs 40 --beta 1 --batch_size 128 --logging_steps 80 --warmup_steps 200 --num_workers 16 --eval_dir_sharpness

python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --layers_per_block $6 --save_dir=/fast/ameterez/nips2024_rebuttal/hessian_spectrum_width --dataset cifar10  --epochs 20 --beta 1 --batch_size 256 --logging_steps 80 --warmup_steps 100 --num_workers 16 --eval_hessian_spectrum --skip_scaling 0