"""
Pytest configuration: add src/pychannel_lab to sys.path so tests can import
core modules without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pychannel_lab"))
