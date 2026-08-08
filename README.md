# Neural-network landscapes at large depth and width

Research code for the experiments accompanying *Super Consistency of Neural
Network Landscapes and Learning Rate Transfer*.

The original PyTorch experiment framework is in [`source_code/`](source_code/),
and plotting/analysis notebooks are in [`notebooks/`](notebooks/).

## Minimal Figure 1 reproduction

[`figure1_jax_reproduction/`](figure1_jax_reproduction/) contains a standalone
JAX reproduction of the width-parameterization comparison from Figure 1. It
removes W&B, PyHessian, ASDL, and the original cluster framework while retaining
the `SimpleConvNet` architecture, explicit parameter scaling, SGD dynamics,
Hessian power iteration, learning-rate sweep, resumable results, and plotting.

See the [reproduction README](figure1_jax_reproduction/README.md) for local and
Slurm commands, including the full 50-epoch CIFAR-10 run.
