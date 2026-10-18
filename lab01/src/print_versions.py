"""Print Python and package versions, save to results/versions.txt."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results" / "versions.txt"


def pkg_names():
    # read names from requirements.txt
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split("==")[0])
    return names


def main():
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"Processor: {platform.processor() or platform.machine()}",
        "",
    ]
    for name in pkg_names():
        try:
            v = version(name)
        except PackageNotFoundError:
            v = "NOT INSTALLED"
        lines.append(f"{name}=={v}")

    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
