# tests/conftest.py
import sys
from pathlib import Path

# Add parent directory to Python path so package imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
