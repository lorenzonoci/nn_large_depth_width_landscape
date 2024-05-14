import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [2]

    # over both
    lrs = np.logspace(-3.0, 1.0, num=20)
    # lrs = np.logspace(0.5, 4.0, num=20)
    # lrs = np.logspace(-1.0, 2.5, num=20)

    parameterizations = ['mup']
    batch_sizes = [256]
    depth_mults = [0]
    width_mults = [4, 8, 16, 32, 64, 128, 256, 512] #x 16
    base_shapes = [64]
    output_mults = [1024, 10000]
    input_mults = [1]

    for run in itertools.product(*[width_mults, depth_mults, seeds, lrs, parameterizations, batch_sizes, base_shapes, output_mults, input_mults]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]} {run[6]} {run[7]} {run[8]}"
        output = f"runs1999/{uid}.stdout"
        error = f"runs1999/{uid}.stderr"
        log = f"runs1999/{uid}.log"
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