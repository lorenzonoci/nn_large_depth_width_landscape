import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [0]
    lrs = np.logspace(-2.5, 1.5, num=19)[:14]

    # over depth 
    # width_mult = 4
    # depth_mults = [1, 2, 4, 8, 16]
    # for run in itertools.product(*[depth_mults, seeds, lrs]):
    #     uid = uuid.uuid4().hex[:10]
    #     arguments = f"{width_mult} {run[0]} {run[1]} {run[2]}"
    #     output = f"runs/{uid}.stdout"
    #     error = f"runs/{uid}.stderr"
    #     log = f"runs/{uid}.log"
    #     cpus = 1
    #     gpus = 1
    #     memory = 10000
    #     disk = "1G"
    #     executable = "run_training.sh"

    #     try:
    #         content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
    #         run_job(uid, 30, content)
    #     except:
    #         raise ValueError("Crashed.")
    # print("Done.")

    # over width
    depth_mult = 1
    width_mults = [2, 4, 8, 16]
    for run in itertools.product(*[width_mults, seeds, lrs]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {depth_mult} {run[1]} {run[2]}"
        output = f"runs/{uid}.stdout"
        error = f"runs/{uid}.stderr"
        log = f"runs/{uid}.log"
        cpus = 17
        gpus = 1
        memory = 10000
        disk = "1G"
        executable = "run_training.sh"

        try:
            content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
            run_job(uid, 40, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")