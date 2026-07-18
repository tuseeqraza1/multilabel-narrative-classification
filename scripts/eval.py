"""Wrapper: run evaluation from the project root (python scripts/eval.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import GOLD_FILE, PRED_FILE, evaluate_files

if __name__ == "__main__":
    evaluate_files(GOLD_FILE, PRED_FILE)
