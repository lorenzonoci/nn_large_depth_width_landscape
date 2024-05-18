#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

# python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --optimizer muadam --save_dir=/home/ameterez/work/nips2024/understanding_adam_wider --dataset cifar10  --epochs 50  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100 --top_hessian_eigvals 1 --eval_hessian

# python main.py  --arch vit --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --save_dir=/home/ameterez/work/nips2024/vit_training_without_lr_scaling --dataset cifar10  --epochs 20 --batch_size 64 --logging_steps 100 --optimizer muadam --num_workers 16 --warmup_steps 500 --top_hessian_eigvals 1 --norm ln --eval_hessian


python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --optimizer muadam --save_dir=/home/ameterez/work/nips2024/understanding_multiple_layers_per_block_longer_with_adam_no_scaling --dataset cifar10  --epochs 50  --layers_per_block $6  --beta 1 --batch_size 256 --logging_steps 100 --warmup_steps 100 --top_hessian_eigvals 1 --eval_hessian
