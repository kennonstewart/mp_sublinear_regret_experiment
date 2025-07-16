{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c0\c93333;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs29\fsmilli14667 \cf0 \expnd0\expndtw0\kerning0
Prompt\
\'a0\
This is your general prompt. I later made a design decision to make the data loader its own submodule within the meta repository, and so the first prompt will be followed by a second that breaks down the process for importing the data for the experiments.\uc0\u8232 \u8232 Experiment Prompt (written prior to meta repository creation)\u8232 \'a0\
Generate a reproducible GitHub repository called `memory-pair-exp`\
that answers the research question:\
\'a0\
\'a0 \'93Does the Memory-Pair learner achieve sub-linear cumulative\
\'a0\'a0 regret R_T = O(\uc0\u8730 T) on drifting and adversarial data streams?\'94\
\'a0\
Repository spec\
---------------\
\'95 Language: Python 3.10\'a0\
\'95 Dependency manager: `pip` with a frozen `requirements.txt`\
\'a0 (torch >= 2.2, numpy, pandas, matplotlib, tqdm, click)\'a0\
\'a0\
Data & Streams\
--------------\
1. Rotating-MNIST: auto-download from\
\'a0\'a0 {\field{\*\fldinst{HYPERLINK "https://github.com/google-research/rotating-mnist"}}{\fldrslt \cf3 \ul \ulc3 https://github.com/google-research/rotating-mnist}}\
2. COVTYPE (UCI Covertype): auto-download from\
\'a0\'a0 {\field{\*\fldinst{HYPERLINK "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/"}}{\fldrslt \cf3 \ul \ulc3 https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/}}\
\'a0\
For each dataset create three streaming generators:\
\'a0 a. IID shuffle\
\'a0 b. Gradual drift (rotation angle +5\'b0 every 1000 steps | feature-mean\
\'a0\'a0\'a0\'a0 drift on COVTYPE)\
\'a0 c. Adversarial permute (random swap every 500 steps)\
\'a0\
Algorithms to implement\
-----------------------\
\'95 MemoryPairOnlineLBFGS\'a0 (our method \'97 single-pass L-BFGS, odometer\
\'a0 disabled for now)\
\'95 OnlineSGD\
\'95 AdaGrad\
\'95 OnlineNewtonStep\'a0 (convex case baseline)\
\'a0\
Regret Evaluation\
-----------------\
* Root script `run_regret.py` takes args:\
\'a0\'a0\'a0 --dataset \{rotmnist,covtype\}\
\'a0\'a0\'a0 --stream\'a0 \{iid,drift,adv\}\
\'a0\'a0\'a0 --algo\'a0\'a0\'a0 \{memorypair,sgd,adagrad,ons\}\
\'a0\'a0\'a0 --T\'a0\'a0\'a0\'a0\'a0\'a0 100000\
\'a0\'a0\'a0 --seed\'a0\'a0\'a0 42\
* Logs cumulative regret to `results/\{dataset\}_\{stream\}_\{algo\}.csv`\
\'a0 (columns: step, regret).\
* Plots log\'96log curve with \uc0\u8730 T guide-line.\
\'a0\
Repro recipe (include in README)\
--------------------------------\
```bash\
git clone {\field{\*\fldinst{HYPERLINK "https://github.com/%3cUSER%3e/memory-pair-exp.git"}}{\fldrslt \cf3 \ul \ulc3 https://github.com/<USER>/memory-pair-exp.git}}\
cd memory-pair-exp\
python -m venv .venv && source .venv/bin/activate\
pip install -r requirements.txt\
python run_regret.py --dataset rotmnist --stream drift --algo memorypair --T 100000\
\'a0\
Protocol for Data Ingestion\
\'a0\
Create/overwrite ONLY the folder\'a0 experiments/sublinear_regret\'a0 with:\
\'a0\
\'a0 README.md\
\'a0 requirements.txt\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 (torch>=2.2, numpy, pandas, matplotlib, click)\
\'a0 run.py\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 (CLI driver)\
\'a0 baselines.py\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 (OnlineSGD, AdaGrad, OnlineNewtonStep)\
\'a0 plotting.py\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0 (helper)\
\'a0 results/.gitkeep\
\'a0\
Data access\
-----------\
Import streams **exclusively** via the shared loader:\
\'a0\
\'a0 from data_loader import (\
\'a0\'a0\'a0\'a0\'a0 get_rotating_mnist_stream,\
\'a0\'a0\'a0\'a0\'a0 get_covtype_stream,\
\'a0 )\
\'a0\
run.py\
------\
CLI\'a0 python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42\
\'a0\
* Resolve Memory-Pair with\
\'a0\'a0\'a0 from code.memory_pair.src.memory_pair import MemoryPair\
* Map dataset+stream flags to the proper generator:\
\'a0\'a0\'a0 if dataset=="rotmnist":\
\'a0\'a0\'a0\'a0\'a0\'a0\'a0 gen = get_rotating_mnist_stream(mode=stream, seed=seed)\
* Compute cumulative regret; save CSV + PNG in results/.\
* At end:\
\'a0\'a0\'a0\'a0\'a0 hash=$(git rev-parse --short HEAD)\
\'a0\'a0\'a0\'a0\'a0 git add results/*\
\'a0\'a0\'a0\'a0\'a0 git commit -m "EXP:sublinear_regret $\{dataset\}-$\{stream\}-$\{algo\} $\{hash\}"\
\'a0\
README.md gives full reproduce command and notes the dependency on\
data_loader and code.memory_pair.\
\'a0\
Do NOT modify files outside experiments/sublinear_regret.\
\uc0\u8232 Prompt used to generate the data loader:\
\uc0\u8232 You are working inside a meta-repository that already contains\
\
  code/memory_pair/\'85           \uc0\u8592  core algorithm\
  experiments/\'85                \uc0\u8592  3 experiment folders (to be filled)\
\
Create a NEW top-level sub-folder called  data_loader\
that will be shared by all experiments.\
\
The sub-module must be entirely self-contained; do not modify\
anything outside  data_loader/.\
\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
FILES & FUNCTIONALITY\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
data_loader/\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  README.md\
\uc0\u9474    \'95 One-paragraph description.\
\uc0\u9474    \'95 Table: dataset key \u8594  loader function.\
\uc0\u9474    \'95 Repro test:  `python sanity_check.py`.\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  requirements.txt      # torchvision, scikit-learn only if available\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  __init__.py\
\uc0\u9474    \'95 from .mnist      import get_rotating_mnist_stream\
\uc0\u9474    \'95 from .cifar10    import get_cifar10_stream\
\uc0\u9474    \'95 from .covtype    import get_covtype_stream\
\uc0\u9474    \'95 from .streams    import make_stream\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  mnist.py\
\uc0\u9474    \'95 download_rotating_mnist(data_dir, split="train")\
\uc0\u9474        \'96 tries torchvision.datasets.MNIST under the hood;\
\uc0\u9474          if import or download fails, calls  _simulate_mnist()\
\uc0\u9474    \'95 get_rotating_mnist_stream(mode, batch_size, seed)\
\uc0\u9474        \'96 mode \u8712  \{"iid","drift","adv"\}\
\uc0\u9474    \'95 _simulate_mnist(n=70000, seed)  # returns numpy arrays (X, y)\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  cifar10.py\
\uc0\u9474    \'95 download_cifar10(data_dir, split)\
\uc0\u9474    \'95 get_cifar10_stream(mode, batch_size, seed)\
\uc0\u9474    \'95 _simulate_cifar10(n=60000, seed)\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  covtype.py\
\uc0\u9474    \'95 download_covtype(data_dir)\
\uc0\u9474    \'95 get_covtype_stream(mode, batch_size, seed)\
\uc0\u9474    \'95 _simulate_covtype(n=581012, d=54, seed)\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  streams.py\
\uc0\u9474    \'95 make_stream(X, y, mode, drift_fn=None, adv_fn=None)\
\uc0\u9474        \'96 generator that yields (x_t, y_t) one at a time\
\uc0\u9474        \'96 if  mode=="drift", applies supplied  drift_fn  every\
\uc0\u9474          1 000 steps; default rotates images or shifts tabular mean\
\uc0\u9474        \'96 if  mode=="adv", adversarially permutes indices every\
\uc0\u9474          500 steps using seed\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  utils.py\
\uc0\u9474    \'95 set_global_seed(seed)\
\uc0\u9474    \'95 download_with_progress(url, target_path)\
\uc0\u9474 \
\uc0\u9492 \u9472 \u9472  sanity_check.py\
    \'95 Command-line script:\
        python sanity_check.py --dataset rotmnist --mode drift --T 5000\
      prints first 5 samples + SHA256 of stream to prove determinism.\
\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
REQUIREMENTS\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
1. **Fail-safe simulation**  \
   If torchvision / internet download fails, _simulate_* functions must\
   generate random but *deterministic* data (use set_global_seed) with\
   identical shapes:  \
     MNIST  \uc0\u8594  28\'d728 uint8, 10 classes  \
     CIFAR10\uc0\u8594  32\'d732\'d73 uint8, 10 classes  \
     COVTYPE\uc0\u8594  float32 tabular with 54 dims, 7 classes\
\
2. **External interface stability**  \
   All experiments will call, e.g.  \
     from data_loader import get_rotating_mnist_stream  \
   Ensure these imports work on fresh clone.\
\
3. **No large binaries**  \
   Simulated data created on-the-fly; real data cached to\
   ~/.cache/memory_pair_data/.\
\
4. **Reproducibility hooks**  \
   Every loader takes a `seed` arg (default 42) and calls\
   utils.set_global_seed(seed).\
\
5. **Do not touch other folders**  \
   Only write files inside  data_loader/.\
\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
GIT\
\uc0\u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \u9472 \
At the end of generation, execute a shell snippet in run-once mode:\
\
```bash\
git add data_loader\
git commit -m "ADD:data_loader \'96 unified loaders w/ fallback simulation"\
\'a0\
}
