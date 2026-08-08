# Minimal Figure 1 reproduction in JAX

This directory is a self-contained, single-seed reproduction of the qualitative
Figure 1 result. It ports the original PyTorch `SimpleConvNet` to pure JAX and
keeps the important experiment semantics:

- raw `N(0, 1)` parameters with explicit fan-in scaling;
- μP readout scaling and width-scaled SGD learning rates;
- NTP scaling and unscaled SGD learning rates;
- top-Hessian-eigenvalue estimates from direct JAX Hessian-vector products;
- fixed-batch losses, a learning-rate sweep, resumable JSON, and plotting.

## Environment

On FASRC, the tested environment is `~/pax`:

```bash
export PAX_PYTHON="$HOME/pax/bin/python"
$PAX_PYTHON -c 'import jax; print(jax.__version__, jax.devices())'
```

The validated stack used Python 3.14.3, JAX/JAXlib 0.9.2 with the CUDA 12
plugin, and an NVIDIA H100 80 GB GPU. To create another environment, install
the local requirements file:

```bash
python -m venv .venv
.venv/bin/pip install -r figure1_jax_reproduction/requirements.txt
```

## Data

The script downloads CIFAR-10 on first use and writes `cifar10_train.npz` in
the directory passed to `--data`. On a cluster, use project or scratch storage
instead of home:

```bash
export CIFAR10_DATA_DIR=/n/netscratch/kempner_sham_lab/Everyone/$USER/cifar10_data
mkdir -p "$CIFAR10_DATA_DIR"
```

It can read the official CIFAR-10 Python archive, a local Hugging Face Parquet
file named `cifar10_train.parquet`, or download `uoft-cs/cifar10` through the
Hugging Face Datasets package.

## Direct smoke test

Run a small test on an allocated GPU, not a login node:

```bash
cd figure1_jax_reproduction
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
${PAX_PYTHON:-$HOME/pax/bin/python} reproduce_jax.py \
  --data "${CIFAR10_DATA_DIR:-cifar10_data}" \
  --out results_smoke --device cuda --widths 8 16 \
  --steps 8 --snapshots 3 --hessian-iters 2 \
  --subset-size 512 --skip-sweep
```

## FASRC Slurm jobs

The supplied scripts request account `kempner_grads`, partition
`kempner_h100`, and one GPU. Submit them from the repository root. Slurm must be
able to open the output directories before the jobs start, so create them first:

```bash
export PAX_PYTHON="$HOME/pax/bin/python"
export CIFAR10_DATA_DIR=/n/netscratch/kempner_sham_lab/Everyone/$USER/cifar10_data

mkdir -p \
  figure1_jax_reproduction/results_fasrc_jax_smoke \
  figure1_jax_reproduction/results_fasrc_jax_full50 \
  figure1_jax_reproduction/results_fasrc_jax_long1024

sbatch figure1_jax_reproduction/slurm_smoke.sh
```

After the smoke test succeeds, submit the full 50-epoch dynamics run:

```bash
sbatch figure1_jax_reproduction/slurm_full50.sh
```

Its exact experiment arguments are:

```text
--subset-size 50000 --widths 64 256 1024 --batch-size 128
--steps 19550 --snapshots 51 --hessian-iters 10 --skip-sweep
```

The intermediate 1,024-step dynamics and learning-rate-sweep run is:

```bash
sbatch figure1_jax_reproduction/slurm_long1024.sh
```

with arguments:

```text
--widths 16 64 256 --subset-size 8192 --batch-size 128
--steps 1024 --snapshots 17 --hessian-iters 10 --sweep-steps 512
```

## Outputs and resuming

Each output directory contains:

- `results.json`: configuration and completed dynamics/sweep results;
- `figure1_minimal.png`: the six-panel sharpness, loss, and LR-transfer plot;
- `environment-<job-id>.txt`: Python, JAX, CUDA, GPU, and exact command;
- `slurm-<job-id>.log`: scheduler output and runtime.

`results.json` is written after every completed model or sweep point. Re-running
the same command with an identical configuration resumes completed work.
