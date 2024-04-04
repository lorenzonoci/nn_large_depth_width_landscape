import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [1]

    # over both
    lrs = np.logspace(-2.5, 2.0, num=20)

    parameterizations = ["mup", 'sp']
    batch_sizes = [256]
    depth_mults = [0]
    width_mults = [1,2,4,8]
    base_shapes = [32, 64, 128]
    # width_mults = [64]

    for run in itertools.product(*[width_mults, depth_mults, seeds, lrs, parameterizations, batch_sizes, base_shapes]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]} {run[6]}"
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
            run_job(uid, 40, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")