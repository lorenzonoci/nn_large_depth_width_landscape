#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --save_dir=/home/ameterez/work/nips2024/top_hessian_many_longer2 --dataset cifar10  --epochs 25  --layers_per_block 1  --beta 1 --skip_scaling 0 --batch_size 256 --logging_steps 100 --warmup_steps 100 --top_hessian_eigvals 10 --eval_hessian