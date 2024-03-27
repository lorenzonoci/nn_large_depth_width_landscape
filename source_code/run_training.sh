#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr mup --save_dir=/home/ameterez/work/icml2024/rebuttal/cifar10_mup_disabled_residuals_2_seeds_rerun --dataset cifar10  --epochs 20  --layers_per_block 1  --beta 1 --batch_size 256 --logging_steps 10 --warmup_steps 100 --skip_scaling 0 --eval_hessian