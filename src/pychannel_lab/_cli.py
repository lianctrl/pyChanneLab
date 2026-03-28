"""
Command-line entry point for pyChanneLab.

Usage (after installing the package):
    pychannelab          # launch Streamlit GUI
    pychannelab --help   # show Streamlit run options
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent / "app.py"
    cmd = ["streamlit", "run", str(app)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
