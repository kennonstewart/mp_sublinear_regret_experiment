import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream

try:
    from torchvision.datasets import CIFAR10
except Exception:
    CIFAR10 = None

DATA_DIR = os.path.expanduser("~/.cache/memory_pair_data/cifar10")


def _simulate_cifar10(n=60000, seed=42):
    set_global_seed(seed)
    X = np.random.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    y = np.random.randint(0, 10, size=(n,), dtype=np.uint8)
    return X, y


def download_cifar10(data_dir=DATA_DIR, split="train"):
    if CIFAR10 is None:
        return _simulate_cifar10(seed=42)
    try:
        dataset = CIFAR10(data_dir, train=(split == "train"), download=True)
        X = dataset.data
        y = np.array(dataset.targets)
        return X, y
    except Exception:
        return _simulate_cifar10(seed=42)


def get_cifar10_stream(mode="iid", batch_size=1, seed=42):
    X, y = download_cifar10()

    def drift_fn(imgs):
        return np.roll(imgs, 1, axis=2)

    def adv_fn(indices):
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices

    return make_stream(X, y, mode=mode, drift_fn=drift_fn, adv_fn=adv_fn, seed=seed)
