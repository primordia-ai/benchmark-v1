from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Climb up the paths until we find the pyproject toml
while not (PROJECT_ROOT / 'pyproject.toml').exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:  # We've reached the root of the filesystem
        raise FileNotFoundError("pyproject.toml not found in any parent directories.")
    PROJECT_ROOT = PROJECT_ROOT.parent
