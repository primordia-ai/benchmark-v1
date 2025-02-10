from pathlib import Path
import os

# Climb up the paths until we find the pyproject toml
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / 'pyproject.toml').exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:  # We've reached the root of the filesystem
        raise FileNotFoundError("pyproject.toml not found in any parent directories.")
    PROJECT_ROOT = PROJECT_ROOT.parent

TESTS_OUT = os.path.join(PROJECT_ROOT, "tests", "tmp")
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")
