import uuid
import itertools
from pycondor.pycondor import *
import numpy as np

if __name__ == '__main__':
    seeds = [0, 1]
    lrs = np.logspace(-2.5, 1.5, num=19)[4:17]

    # over width
    depth_mult = 1
    width_mults = [2, 4, 8, 16, 32]
    params = ['mup']
    for run in itertools.product(*[width_mults, seeds, lrs, params]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{run[0]} {depth_mult} {run[1]} {run[2]} {run[3]}"
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
            run_job(uid, 300, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")