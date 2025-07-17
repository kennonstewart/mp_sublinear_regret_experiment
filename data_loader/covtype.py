import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream

try:
    from sklearn.datasets import fetch_covtype
except Exception:
    fetch_covtype = None

DATA_DIR = os.path.expanduser("~/.cache/memory_pair_data/covtype")


def _simulate_covtype(n=581012, d=54, seed=42):
    set_global_seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, 7, size=(n,), dtype=np.int64)
    return X, y


def download_covtype(data_dir=DATA_DIR):
    if fetch_covtype is None:
        return _simulate_covtype(seed=42)
    try:
        data = fetch_covtype(data_home=data_dir)
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        return X, y
    except Exception:
        return _simulate_covtype(seed=42)


def get_covtype_stream(mode="iid", batch_size=1, seed=42):
    X, y = download_covtype()

    def drift_fn(tab):
        return tab + 0.1 * np.random.randn(*tab.shape)

    def adv_fn(indices):
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices

    return make_stream(X, y, mode=mode, drift_fn=drift_fn, adv_fn=adv_fn, seed=seed)
