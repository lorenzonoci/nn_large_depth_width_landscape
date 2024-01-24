#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mup

python main.py  --arch conv --width_mult $1 --depth_mult $2 --seed $3 --lr $4 --parametr mup_sqrt_depth --save_dir=/lustre/home/ameterez/work/icml2024/results --dataset cifar10  --epochs 50  --layers_per_block 1  --beta 3 --batch_size 512 --logging_steps 200 --warmup_steps 100 --no_data_augm