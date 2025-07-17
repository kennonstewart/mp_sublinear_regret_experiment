import numpy as np
from .utils import set_global_seed

def make_stream(X, y, mode="iid", drift_fn=None, adv_fn=None, seed=42):
    set_global_seed(seed)
    n = len(X)
    indices = np.arange(n)
    step = 0
    while True:
        i = indices[step % n]
        yield X[i], y[i]
        step += 1
        if mode == "drift" and step % 1000 == 0 and drift_fn is not None:
            X = drift_fn(X)
        if mode == "adv" and step % 500 == 0 and adv_fn is not None:
            indices = adv_fn(indices)
