import click
from . import get_rotating_mnist_stream, get_covtype_stream
from .utils import sha256_of_stream

DATASET_MAP = {
    "rotmnist": get_rotating_mnist_stream,
    "covtype": get_covtype_stream,
}

@click.command()
@click.option("--dataset", type=click.Choice(list(DATASET_MAP.keys())), required=True)
@click.option("--mode", type=click.Choice(["iid", "drift", "adv"]), default="iid")
@click.option("--t", type=int, default=5000)
def main(dataset, mode, t):
    gen = DATASET_MAP[dataset](mode=mode, seed=42)
    for i, sample in zip(range(5), gen):
        print(sample)
    sha = sha256_of_stream(gen, T=t)
    print("SHA256", sha)

if __name__ == "__main__":
    main()
