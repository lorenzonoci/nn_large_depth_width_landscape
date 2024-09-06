import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [0, 1]
    # lrs = np.logspace(-2, 0.5, num=16) #sgd
    # lrs = np.logspace(-1, 0, num=14) #sgd
    lrs = np.logspace(-1, 0, num=15)[4:10]
    # lrs = np.logspace(-1, 1, num=18)[0:-3]
#    lrs = np.logspace(-1, 1.5, num=24)[:-9]
    # lrs = np.logspace(1.5, 4.0, num=24)[:10]
    # lrs = np.logspace(-3, -1, num=12) #adam with scaling
    # lrs = np.logspace(-5, -2.5, num=15) # adam without scaling
    # over width
    #depth_mults = [64, 32, 16, 8, 4, 2, 1]
    depth_mults = [1]
    width_mults = [1, 2, 4, 8, 16]
    # width_mults = [4]
    layers_per_blocks = [1]
    params = ['mup']
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
            run_job(uid, 100, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")
