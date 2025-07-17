import os
import random
import hashlib
import urllib.request

import numpy as np


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def download_with_progress(url: str, target_path: str):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        return target_path
    with urllib.request.urlopen(url) as response, open(target_path, 'wb') as out:
        data = response.read()
        out.write(data)
    return target_path


def sha256_of_stream(generator, T=1000):
    h = hashlib.sha256()
    for i, (x, y) in enumerate(generator):
        h.update(x.tobytes())
        h.update(bytes([int(y)]))
        if i + 1 >= T:
            break
    return h.hexdigest()
