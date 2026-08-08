#!/usr/bin/env python3
"""JAX reproduction of the qualitative result in Figure 1.

This is a functional port of reproduce.py.  It uses raw N(0, 1) parameters,
the same explicit fan-in scalings, SGD updates, direct Hessian-vector products,
and the same output schema.  CIFAR-10 is downloaded and decoded without
PyTorch/torchvision so the script can run in the existing ~/pax environment.
"""

import argparse
import gc
import io
import json
import math
import os
import pickle
import random
import tarfile
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def select_device(requested):
    backend = jax.default_backend()
    if requested == "cuda" and backend != "gpu":
        raise RuntimeError(
            f"CUDA was requested but JAX selected backend={backend!r}; "
            f"devices={jax.devices()}"
        )
    if requested == "cpu" and backend != "cpu":
        raise RuntimeError(
            "To force CPU, launch with JAX_PLATFORMS=cpu before importing JAX"
        )
    return backend, jax.devices()


def download_cifar10(data_dir):
    """Download CIFAR-10 and cache a compact NumPy representation."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "cifar10_train.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["images"], cached["labels"]

    parquet_path = data_dir / "cifar10_train.parquet"
    if parquet_path.exists():
        import pyarrow.parquet as parquet
        from PIL import Image

        print(f"decoding {parquet_path}", flush=True)
        table = parquet.read_table(parquet_path, columns=["img", "label"])
        encoded_images = table["img"].to_pylist()
        all_images = np.stack(
            [
                np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"), dtype=np.uint8)
                for item in encoded_images
            ]
        )
        all_labels = np.asarray(table["label"].to_numpy(), dtype=np.int64)
        np.savez(cache_path, images=all_images, labels=all_labels)
        return all_images, all_labels

    archive_path = data_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        try:
            from datasets import load_dataset

            print(f"downloading uoft-cs/cifar10 to {data_dir}", flush=True)
            dataset = load_dataset(
                "uoft-cs/cifar10",
                split="train",
                cache_dir=str(data_dir / "huggingface"),
            )
            image_column = "img" if "img" in dataset.column_names else "image"
            all_images = np.stack(
                [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in dataset[image_column]]
            )
            all_labels = np.asarray(dataset["label"], dtype=np.int64)
            np.savez(cache_path, images=all_images, labels=all_labels)
            return all_images, all_labels
        except Exception as error:
            print(f"Hugging Face download failed ({error!r}); trying {CIFAR_URL}", flush=True)
            partial_path = archive_path.with_suffix(archive_path.suffix + ".partial")
            urllib.request.urlretrieve(CIFAR_URL, partial_path)
            partial_path.replace(archive_path)

    images = []
    labels = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for batch_index in range(1, 6):
            name = f"cifar-10-batches-py/data_batch_{batch_index}"
            member = archive.getmember(name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {name} from {archive_path}")
            batch = pickle.load(extracted, encoding="bytes")
            batch_images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            images.append(batch_images.astype(np.uint8, copy=False))
            labels.append(np.asarray(batch[b"labels"], dtype=np.int64))

    all_images = np.concatenate(images)
    all_labels = np.concatenate(labels)
    np.savez(cache_path, images=all_images, labels=all_labels)
    return all_images, all_labels


def make_data(data_dir, subset_size, seed):
    images, labels = download_cifar10(data_dir)
    if subset_size > len(images):
        raise ValueError(f"subset-size {subset_size} exceeds CIFAR-10 size {len(images)}")
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(images))[:subset_size]
    return images[ids], labels[ids]


def normalize(images):
    return images.astype(np.float32) / 127.5 - 1.0


def augment(images, rng):
    batch_size = len(images)
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="constant")
    row_offsets = rng.integers(0, 9, size=batch_size)
    col_offsets = rng.integers(0, 9, size=batch_size)
    rows = row_offsets[:, None] + np.arange(32)[None, :]
    cols = col_offsets[:, None] + np.arange(32)[None, :]
    cropped = padded[
        np.arange(batch_size)[:, None, None],
        rows[:, :, None],
        cols[:, None, :],
        :,
    ].copy()
    flip = rng.random(batch_size) < 0.5
    cropped[flip] = cropped[flip, :, ::-1, :]
    return normalize(cropped)


class BatchStream:
    def __init__(self, images, labels, batch_size, seed):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.offset = 0

    def __next__(self):
        if self.offset >= len(self.order):
            self.order = self.rng.permutation(len(self.images))
            self.offset = 0
        ids = self.order[self.offset : self.offset + self.batch_size]
        self.offset += self.batch_size
        return augment(self.images[ids], self.rng), self.labels[ids]


def init_params(seed, width):
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    return {
        "conv1": jax.random.normal(key1, (3, 3, 3, width), dtype=jnp.float32),
        "conv2": jax.random.normal(key2, (3, 3, width, width), dtype=jnp.float32),
        "readout": jax.random.normal(key3, (width, 10), dtype=jnp.float32),
    }


def parameter_scales(width, parametrization):
    gamma = math.sqrt(width) if parametrization == "mup" else 1.0
    return (
        1.0 / math.sqrt(3 * 3 * 3),
        1.0 / math.sqrt(3 * 3 * width),
        1.0 / math.sqrt(width) / gamma,
    )


def max_pool2d(x):
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )


def forward(params, x, scales):
    scale1, scale2, scale3 = scales
    x = jax.lax.conv_general_dilated(
        x,
        params["conv1"] * scale1,
        window_strides=(2, 2),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = max_pool2d(jax.nn.relu(x))
    x = jax.lax.conv_general_dilated(
        x,
        params["conv2"] * scale2,
        window_strides=(4, 4),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = max_pool2d(jax.nn.relu(x))
    return x.reshape((x.shape[0], -1)) @ (params["readout"] * scale3)


def cross_entropy(params, x, y, scales):
    logits = forward(params, x, scales)
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(len(y)), y])


@jax.jit
def train_step(params, x, y, scales, learning_rate):
    loss, grads = jax.value_and_grad(cross_entropy)(params, x, y, scales)
    updated = jax.tree_util.tree_map(
        lambda param, grad: param - learning_rate * grad, params, grads
    )
    return updated, loss


loss_on = jax.jit(cross_entropy)
grad_loss = jax.grad(cross_entropy)


@jax.jit
def hessian_vector_product(params, vector, x, y, scales):
    return jax.jvp(
        lambda candidate: grad_loss(candidate, x, y, scales),
        (params,),
        (vector,),
    )[1]


def tree_dot(left, right):
    products = jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), left, right)
    return sum(jax.tree_util.tree_leaves(products))


def tree_normalize(tree):
    norm = jnp.sqrt(tree_dot(tree, tree)).clip(min=1e-12)
    return jax.tree_util.tree_map(lambda value: value / norm, tree)


def random_vector_like(params, seed):
    leaves, structure = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    vector = [jax.random.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)]
    return tree_normalize(jax.tree_util.tree_unflatten(structure, vector))


def top_hessian_eigenvalue(params, x, y, scales, iterations, vector=None, seed=0):
    if vector is None:
        vector = random_vector_like(params, seed)
    eigenvalue = jnp.asarray(0.0)
    for _ in range(iterations):
        hv = hessian_vector_product(params, vector, x, y, scales)
        eigenvalue = tree_dot(vector, hv)
        vector = tree_normalize(hv)
    return abs(float(eigenvalue)), vector


def clear_device():
    gc.collect()


def train_dynamics(args, train_images, train_labels, eval_batch, hessian_batch, parametrization, width):
    seed_everything(args.seed)
    params = init_params(args.seed, width)
    scales = parameter_scales(width, parametrization)
    base_lr = args.mup_lr if parametrization == "mup" else args.ntp_lr
    physical_lr = base_lr * width if parametrization == "mup" else base_lr
    stream = BatchStream(train_images, train_labels, args.batch_size, args.seed)
    checkpoints = set(np.unique(np.linspace(0, args.steps, args.snapshots).round().astype(int)))
    result = {"step": [], "epoch": [], "loss": [], "sharpness": []}
    eigenvector = None
    steps_per_epoch = math.ceil(len(train_images) / args.batch_size)

    for step in range(args.steps + 1):
        if step in checkpoints:
            value, eigenvector = top_hessian_eigenvalue(
                params,
                *hessian_batch,
                scales,
                args.hessian_iters,
                eigenvector,
                args.seed,
            )
            if parametrization == "mup":
                value *= width
            current_loss = float(loss_on(params, *eval_batch, scales))
            result["step"].append(int(step))
            result["epoch"].append(step / steps_per_epoch)
            result["loss"].append(current_loss)
            result["sharpness"].append(value)
            print(
                f"{parametrization:>3} width={width:>4} step={step:>5}/{args.steps} "
                f"loss={current_loss:.3f} sharpness={value:.3g}",
                flush=True,
            )
        if step == args.steps:
            break
        x, y = next(stream)
        params, _ = train_step(params, jnp.asarray(x), jnp.asarray(y), scales, physical_lr)

    del params, eigenvector
    clear_device()
    return result


def train_final_loss(args, train_images, train_labels, eval_batch, parametrization, width, base_lr):
    seed_everything(args.seed)
    params = init_params(args.seed, width)
    scales = parameter_scales(width, parametrization)
    physical_lr = base_lr * width if parametrization == "mup" else base_lr
    stream = BatchStream(train_images, train_labels, args.batch_size, args.seed)
    failed = False
    for _ in range(args.sweep_steps):
        x, y = next(stream)
        params, loss = train_step(params, jnp.asarray(x), jnp.asarray(y), scales, physical_lr)
        if not math.isfinite(float(loss)):
            failed = True
            break
    final_loss = 10.0 if failed else min(float(loss_on(params, *eval_batch, scales)), 10.0)
    print(
        f"sweep {parametrization:>3} width={width:>4} lr={base_lr:>7g} "
        f"loss={final_loss:.3f}",
        flush=True,
    )
    del params
    clear_device()
    return final_loss


def save(cache_path, results, config):
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"config": config, "results": results}, indent=2))
    temporary.replace(cache_path)


def plot(results, args, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(args.widths)))
    for row, parametrization in enumerate(("mup", "ntp")):
        label = r"$\mu$P" if parametrization == "mup" else "NTP"
        for color, width in zip(colors, args.widths):
            run = results[f"dynamic:{parametrization}:{width}"]
            axes[row, 0].plot(run["epoch"], run["sharpness"], color=color, label=str(width))
            axes[row, 1].plot(run["epoch"], run["loss"], color=color)
        learning_rate = args.mup_lr if parametrization == "mup" else args.ntp_lr
        axes[row, 0].axhline(2 / learning_rate, color="0.25", ls="--", lw=1, label="2 / lr")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(f"{label} — sharpness")
        axes[row, 1].set_title(f"{label} — loss")
        axes[row, 0].set_ylabel(r"$\lambda_{\max}$")

        if not args.skip_sweep:
            sweep_lrs = args.mup_sweep if parametrization == "mup" else args.ntp_sweep
            for color, width in zip(colors, args.widths):
                values = [results[f"sweep:{parametrization}:{width}:{lr:g}"] for lr in sweep_lrs]
                axes[row, 2].plot(sweep_lrs, values, "o-", color=color)
            axes[row, 2].set_xscale("log")
            axes[row, 2].set_title(f"{label} — LR transfer")
        else:
            axes[row, 2].text(0.5, 0.5, "sweep skipped", ha="center", va="center")

    for axis in axes[:, 0]:
        axis.set_xlabel("Epoch on subset")
        axis.grid(alpha=0.25)
    for axis in axes[:, 1]:
        axis.set_xlabel("Epoch on subset")
        axis.set_ylabel("Fixed-batch loss")
        axis.grid(alpha=0.25)
    for axis in axes[:, 2]:
        axis.set_xlabel("Base learning rate")
        axis.set_ylabel("Final fixed-batch loss")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(title="Width", ncol=2, fontsize=8)
    fig.suptitle("Minimal CIFAR-10 reproduction of Figure 1 (JAX, one seed)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results_jax"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--subset-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--snapshots", type=int, default=9)
    parser.add_argument("--hessian-batch", type=int, default=32)
    parser.add_argument("--hessian-iters", type=int, default=8)
    parser.add_argument("--mup-lr", type=float, default=0.5)
    parser.add_argument("--ntp-lr", type=float, default=7.0)
    parser.add_argument("--sweep-steps", type=int, default=128)
    parser.add_argument("--mup-sweep", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0])
    parser.add_argument("--ntp-sweep", type=float, nargs="+", default=[1.0, 3.0, 7.0, 20.0, 60.0])
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    backend, devices = select_device(args.device)
    print(f"jax={jax.__version__} backend={backend} devices={devices}", flush=True)
    train_images, train_labels = make_data(args.data, args.subset_size, args.seed)
    eval_count = min(512, len(train_images))
    eval_batch = (
        jnp.asarray(normalize(train_images[:eval_count])),
        jnp.asarray(train_labels[:eval_count]),
    )
    hessian_batch = (
        eval_batch[0][: args.hessian_batch],
        eval_batch[1][: args.hessian_batch],
    )

    config = vars(args).copy()
    config["data"] = str(config["data"])
    config["out"] = str(config["out"])
    config["implementation"] = "jax"
    config["jax_version"] = jax.__version__
    cache_path = args.out / "results.json"
    results = {}
    if cache_path.exists():
        old = json.loads(cache_path.read_text())
        if old.get("config") == config:
            results = old["results"]
            print(f"resuming {len(results)} completed runs from {cache_path}", flush=True)

    for parametrization in ("mup", "ntp"):
        for width in args.widths:
            key = f"dynamic:{parametrization}:{width}"
            if key not in results:
                results[key] = train_dynamics(
                    args,
                    train_images,
                    train_labels,
                    eval_batch,
                    hessian_batch,
                    parametrization,
                    width,
                )
                save(cache_path, results, config)

    if not args.skip_sweep:
        for parametrization, learning_rates in (("mup", args.mup_sweep), ("ntp", args.ntp_sweep)):
            for width in args.widths:
                for learning_rate in learning_rates:
                    key = f"sweep:{parametrization}:{width}:{learning_rate:g}"
                    if key not in results:
                        results[key] = train_final_loss(
                            args,
                            train_images,
                            train_labels,
                            eval_batch,
                            parametrization,
                            width,
                            learning_rate,
                        )
                        save(cache_path, results, config)

    figure_path = args.out / "figure1_minimal.png"
    plot(results, args, figure_path)
    print(f"wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
