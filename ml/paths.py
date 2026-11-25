from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = 42

for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
