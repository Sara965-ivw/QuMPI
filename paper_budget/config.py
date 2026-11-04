"""Single source of truth for paths, seeds and budgets (paper_budget)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the QuMPI repository folder
DATA_V2 = ROOT / "data" / "corpus_v2"                 # frozen thesis data (read only)
DATA_V4 = ROOT / "data" / "corpus_v4"                 # new data for the paper
RUNS = ROOT / "runs" / "paper_budget"

DEVICE = "fake_brisbane"
TRANSPILER_SEEDS = (1234, 2027, 3141)                  # 1234 = thesis seed, two extra
TRAIN_SEEDS = (0, 1, 2, 3, 4)                          # five models = the ensemble
BUDGETS = (1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)          # shots N at which MSE is judged
SPLIT_SEED = 2026
