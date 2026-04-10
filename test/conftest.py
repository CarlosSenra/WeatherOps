from pathlib import Path
import sys


# Ensure repository root is importable for tests in any subfolder.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
