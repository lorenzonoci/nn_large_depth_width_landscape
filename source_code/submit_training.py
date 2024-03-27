import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [2, 10]
    lrs = np.logspace(-2.5, 1.5, num=19)[:14]

    # over both
    depth_mults = [0, 1, 2]
    width_mults = [2, 4]
    for run in itertools.product(*[width_mults, depth_mults, seeds, lrs]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {run[1]} {run[2]} {run[3]}"
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
            run_job(uid, 80, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")