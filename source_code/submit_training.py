import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [74]
    # lrs = np.logspace(-2.5, 1.5, num=19)[4:17] #sgd
    # lrs = np.logspace(-3, -1, num=12) #adam with scaling
    lrs = np.logspace(-4, -2, num=12) # adam without scaling

    # over width
    depth_mults = [16,8,4,2,1]
    width_mults = [0.5]
    layers_per_blocks = [1, 2]
    # layers_per_blocks = [4]
    params = ['mup_sqrt_depth']
    for run in itertools.product(*[width_mults, depth_mults, seeds, lrs, params, layers_per_blocks]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]}"
        output = f"runs/{uid}.stdout"
        error = f"runs/{uid}.stderr"
        log = f"runs/{uid}.log"
        cpus = 17
        gpus = 1
        memory = 40000
        disk = "1G"
        executable = "run_training.sh"

        try:
            content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
            run_job(uid, 150, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")