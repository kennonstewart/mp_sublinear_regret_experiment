import os
import numpy as np
from .utils import set_global_seed, download_with_progress
from .streams import make_stream

try:
    from torchvision.datasets import MNIST
except Exception:
    MNIST = None

DATA_DIR = os.path.expanduser("~/.cache/memory_pair_data/mnist")


def _simulate_mnist(n=70000, seed=42):
    set_global_seed(seed)
    X = np.random.randint(0, 256, size=(n, 28, 28), dtype=np.uint8)
    y = np.random.randint(0, 10, size=(n,), dtype=np.uint8)
    return X, y


def download_rotating_mnist(data_dir=DATA_DIR, split="train"):
    if MNIST is None:
        return _simulate_mnist(seed=42)
    try:
        dataset = MNIST(data_dir, train=(split == "train"), download=True)
        X = dataset.data.numpy()
        y = dataset.targets.numpy()
        return X, y
    except Exception:
        return _simulate_mnist(seed=42)


def get_rotating_mnist_stream(mode="iid", batch_size=1, seed=42):
    X, y = download_rotating_mnist()

    def drift_fn(imgs):
        return np.rot90(imgs, k=1, axes=(1, 2))

    def adv_fn(indices):
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices

    return make_stream(X, y, mode=mode, drift_fn=drift_fn, adv_fn=adv_fn, seed=seed)
