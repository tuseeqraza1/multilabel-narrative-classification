"""Wrapper: run training from the project root (python scripts/train.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training import main

if __name__ == "__main__":
    main()
