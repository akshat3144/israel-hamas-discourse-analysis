"""Run all RQ4 toxicity scripts in order."""

from pathlib import Path
import runpy


SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"


def main():
    scripts = [
        "toxicity_assessment.py",
    ]

    for script in scripts:
        script_path = SCRIPT_DIR / script
        print(f"\n=== Running {script_path} ===")
        runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
