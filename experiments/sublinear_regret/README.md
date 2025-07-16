# Sublinear Regret Experiment

This experiment answers the research question: "Does the Memory-Pair learner achieve sub-linear cumulative regret R_T = O(√T) on drifting and adversarial data streams?"

## Dependencies

This experiment depends on:
- `data_loader` module for dataset streaming
- `code.memory_pair` module for the core Memory-Pair algorithm

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment
python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42
```

### Arguments

- `--dataset`: Dataset to use (`rotmnist`, `covtype`)
- `--stream`: Stream type (`iid`, `drift`, `adv`)
- `--algo`: Algorithm (`memorypair`, `sgd`, `adagrad`, `ons`)
- `--T`: Number of time steps (default: 100000)
- `--seed`: Random seed (default: 42)

## Output

Results are saved in the `results/` directory:
- CSV file: `{dataset}_{stream}_{algo}.csv` with columns (step, regret)
- PNG plot: `{dataset}_{stream}_{algo}.png` with log-log curve and √T guide-line

## Algorithms

- **memorypair**: Memory-Pair Online L-BFGS (our method)
- **sgd**: Online Stochastic Gradient Descent
- **adagrad**: Adaptive Gradient Algorithm
- **ons**: Online Newton Step (convex baseline)