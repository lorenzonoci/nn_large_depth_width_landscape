#!/usr/bin/env bash
#SBATCH --job-name=fig1-jax-long1024
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=figure1_jax_reproduction/results_fasrc_jax_long1024/slurm-%j.log

set -euo pipefail

PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="$PROJECT/results_fasrc_jax_long1024"
DATA=${CIFAR10_DATA_DIR:-$PROJECT/cifar10_data}
PYTHON=${PAX_PYTHON:-$HOME/pax/bin/python}

mkdir -p "$OUT" "$DATA" "$PROJECT/.cache/matplotlib" "$PROJECT/.jax_cache"
cd "$PROJECT"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib"
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="$PROJECT/.jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

{
  date --iso-8601=seconds
  hostname
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "slurm_partition=$SLURM_JOB_PARTITION"
  echo "command=$PYTHON reproduce_jax.py --data $DATA --out $OUT --device cuda --steps 1024 --snapshots 17 --hessian-iters 10 --sweep-steps 512"
  nvidia-smi
  "$PYTHON" -c 'import platform, jax, jaxlib; print("python=" + platform.python_version()); print("jax=" + jax.__version__); print("jaxlib=" + jaxlib.__version__); print("backend=" + jax.default_backend()); print("devices=" + str(jax.devices()))'
} 2>&1 | tee "$OUT/environment-$SLURM_JOB_ID.txt"

start_seconds=$(date +%s)
"$PYTHON" "$PROJECT/reproduce_jax.py" \
  --data "$DATA" \
  --out "$OUT" \
  --device cuda \
  --steps 1024 \
  --snapshots 17 \
  --hessian-iters 10 \
  --sweep-steps 512
end_seconds=$(date +%s)
echo "runtime_seconds=$((end_seconds - start_seconds))"
ls -lh "$OUT"
