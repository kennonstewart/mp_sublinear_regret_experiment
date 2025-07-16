Prompt

This is your general prompt. I later made a design decision to make the data loader its own submodule within the meta repository, and so the first prompt will be followed by a second that breaks down the process for importing the data for the experiments.

Experiment Prompt (written prior to meta repository creation)

Generate a reproducible GitHub repository called `memory-pair-exp`
that answers the research question:

  “Does the Memory-Pair learner achieve sub-linear cumulative
   regret R_T = O(√T) on drifting and adversarial data streams?”

Repository spec
---------------
• Language: Python 3.10  
• Dependency manager: `pip` with a frozen `requirements.txt`
  (torch >= 2.2, numpy, pandas, matplotlib, tqdm, click)  

Data & Streams
--------------
1. Rotating-MNIST: auto-download from
   https://github.com/google-research/rotating-mnist
2. COVTYPE (UCI Covertype): auto-download from
   https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/

For each dataset create three streaming generators:
  a. IID shuffle
  b. Gradual drift (rotation angle +5° every 1000 steps | feature-mean
     drift on COVTYPE)
  c. Adversarial permute (random swap every 500 steps)

Algorithms to implement
-----------------------
• MemoryPairOnlineLBFGS  (our method — single-pass L-BFGS, odometer
  disabled for now)
• OnlineSGD
• AdaGrad
• OnlineNewtonStep  (convex case baseline)

Regret Evaluation
-----------------
* Root script `run_regret.py` takes args:
    --dataset {rotmnist,covtype}
    --stream  {iid,drift,adv}
    --algo    {memorypair,sgd,adagrad,ons}
    --T       100000
    --seed    42
* Logs cumulative regret to `results/{dataset}_{stream}_{algo}.csv`
  (columns: step, regret).
* Plots log–log curve with √T guide-line.

Repro recipe (include in README)
--------------------------------
```bash
git clone https://github.com/<USER>/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_regret.py --dataset rotmnist --stream drift --algo memorypair --T 100000

Protocol for Data Ingestion

Create/overwrite ONLY the folder  experiments/sublinear_regret  with:

  README.md
  requirements.txt          (torch>=2.2, numpy, pandas, matplotlib, click)
  run.py                    (CLI driver)
  baselines.py              (OnlineSGD, AdaGrad, OnlineNewtonStep)
  plotting.py               (helper)
  results/.gitkeep

Data access
-----------
Import streams **exclusively** via the shared loader:

  from data_loader import (
      get_rotating_mnist_stream,
      get_covtype_stream,
  )

run.py
------
CLI  python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42

* Resolve Memory-Pair with
    from code.memory_pair.src.memory_pair import MemoryPair
* Map dataset+stream flags to the proper generator:
    if dataset=="rotmnist":
        gen = get_rotating_mnist_stream(mode=stream, seed=seed)
* Compute cumulative regret; save CSV + PNG in results/.
* At end:
      hash=$(git rev-parse --short HEAD)
      git add results/*
      git commit -m "EXP:sublinear_regret ${dataset}-${stream}-${algo} ${hash}"

README.md gives full reproduce command and notes the dependency on
data_loader and code.memory_pair.

Do NOT modify files outside experiments/sublinear_regret.

Prompt used to generate the data loader:


You are working inside a meta-repository that already contains

  code/memory_pair/…           ← core algorithm
  experiments/…                ← 3 experiment folders (to be filled)

Create a NEW top-level sub-folder called  data_loader
that will be shared by all experiments.

The sub-module must be entirely self-contained; do not modify
anything outside  data_loader/.

─────────────────────────────────────────────────────────────────
FILES & FUNCTIONALITY
─────────────────────────────────────────────────────────────────
data_loader/
│
├── README.md
│   • One-paragraph description.
│   • Table: dataset key → loader function.
│   • Repro test:  `python sanity_check.py`.
│
├── requirements.txt      # torchvision, scikit-learn only if available
│
├── __init__.py
│   • from .mnist      import get_rotating_mnist_stream
│   • from .cifar10    import get_cifar10_stream
│   • from .covtype    import get_covtype_stream
│   • from .streams    import make_stream
│
├── mnist.py
│   • download_rotating_mnist(data_dir, split="train")
│       – tries torchvision.datasets.MNIST under the hood;
│         if import or download fails, calls  _simulate_mnist()
│   • get_rotating_mnist_stream(mode, batch_size, seed)
│       – mode ∈ {"iid","drift","adv"}
│   • _simulate_mnist(n=70000, seed)  # returns numpy arrays (X, y)
│
├── cifar10.py
│   • download_cifar10(data_dir, split)
│   • get_cifar10_stream(mode, batch_size, seed)
│   • _simulate_cifar10(n=60000, seed)
│
├── covtype.py
│   • download_covtype(data_dir)
│   • get_covtype_stream(mode, batch_size, seed)
│   • _simulate_covtype(n=581012, d=54, seed)
│
├── streams.py
│   • make_stream(X, y, mode, drift_fn=None, adv_fn=None)
│       – generator that yields (x_t, y_t) one at a time
│       – if  mode=="drift", applies supplied  drift_fn  every
│         1 000 steps; default rotates images or shifts tabular mean
│       – if  mode=="adv", adversarially permutes indices every
│         500 steps using seed
│
├── utils.py
│   • set_global_seed(seed)
│   • download_with_progress(url, target_path)
│
└── sanity_check.py
    • Command-line script:
        python sanity_check.py --dataset rotmnist --mode drift --T 5000
      prints first 5 samples + SHA256 of stream to prove determinism.

─────────────────────────────────────────────────────────────────
REQUIREMENTS
─────────────────────────────────────────────────────────────────
1. **Fail-safe simulation**  
   If torchvision / internet download fails, _simulate_* functions must
   generate random but *deterministic* data (use set_global_seed) with
   identical shapes:  
     MNIST  → 28×28 uint8, 10 classes  
     CIFAR10→ 32×32×3 uint8, 10 classes  
     COVTYPE→ float32 tabular with 54 dims, 7 classes

2. **External interface stability**  
   All experiments will call, e.g.  
     from data_loader import get_rotating_mnist_stream  
   Ensure these imports work on fresh clone.

3. **No large binaries**  
   Simulated data created on-the-fly; real data cached to
   ~/.cache/memory_pair_data/.

4. **Reproducibility hooks**  
   Every loader takes a `seed` arg (default 42) and calls
   utils.set_global_seed(seed).

5. **Do not touch other folders**  
   Only write files inside  data_loader/.

─────────────────────────────────────────────────────────────────
GIT
─────────────────────────────────────────────────────────────────
At the end of generation, execute a shell snippet in run-once mode:

```bash
git add data_loader
git commit -m "ADD:data_loader – unified loaders w/ fallback simulation"
<img width="1192" height="7306" alt="image" src="https://github.com/user-attachments/assets/916ec91a-23f8-466f-9ef7-179988c7fdf1" />
