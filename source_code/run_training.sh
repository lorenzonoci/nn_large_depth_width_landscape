#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --batch_size $6 --base_shape $7 --save_dir=/home/ameterez/work/icml2024/rebuttal/base_shape_ablation --dataset cifar10 --epochs 50 --logging_steps 100 --warmup_steps 100 --eval_hessian