#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

python main.py --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr $5 --batch_size $6 --base_shape $7 --output_mult $8 --input_mult $9 --save_dir=/home/ameterez/work/icml2024/rebuttal/base_shape_ablation_base_shape_32_with_small_output_multiplier_input_smaller_net_relu2 --dataset cifar10 --epochs 50 --logging_steps 100 --warmup_steps 500 --eval_hessian