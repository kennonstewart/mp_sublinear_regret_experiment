# Data Loader

Shared dataset loaders with deterministic fallbacks. Available loaders:

| key      | function                      |
|----------|-------------------------------|
| rotmnist | `get_rotating_mnist_stream`   |
| cifar10  | `get_cifar10_stream`          |
| covtype  | `get_covtype_stream`          |

Run `python sanity_check.py` to verify deterministic streams.
