"""Run the data preparation and EDA scripts for this folder."""

from pathlib import Path
import runpy


SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"


def main():
    scripts = [
        "eda_analysis.py",
    ]

    for script in scripts:
        script_path = SCRIPT_DIR / script
        print(f"\n=== Running {script_path} ===")
        runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
