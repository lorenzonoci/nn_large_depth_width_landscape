import uuid
import itertools
import subprocess
import numpy as np
import time

if __name__ == '__main__':

    # over both
    lrs = np.logspace(-1.5, 0.5, num=8)

    widths = np.arange(2, 37, 9)[1:2]
    seeds = [0]
    gpus = [5]
    epochs = 20
    batch_size = 128
    logging_steps = 50
    
    for i, run in enumerate(itertools.product(*[lrs, seeds, widths])):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {run[1]} {run[2]}"
        gpu = gpus[i % len(gpus)]
        print(f'{i}:{arguments}')
        try:
            cmd = f'tmux new-session -d -s "{i}" "CUDA_VISIBLE_DEVICES={gpu} python source_code/main.py --optimizer musgd \
                --arch simple_conv --lr {run[0]} --width {run[2]} --parametr mup \
                --save_dir=/local/home/lnoci/nn_large_depth_width_landscape-1/experiments/convergence_optimal_lr/ \
                --dataset cifar10  --epochs {epochs} --batch_size {batch_size}  --warmup_steps 0 --seed {run[1]}  \
                --logging_steps {logging_steps}"'
            # --eval_hessian --top_hessian_eigvals 10 
            print(cmd)
            subprocess.run(cmd, shell=True)
        except:
            raise ValueError("Crashed.")
        time.sleep(1)
    print("Done.")