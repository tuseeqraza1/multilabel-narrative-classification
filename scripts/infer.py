"""Wrapper: run inference from the project root (python scripts/infer.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import main

if __name__ == "__main__":
    main()
