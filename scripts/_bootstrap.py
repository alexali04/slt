"""Put ``src/`` on the path so scripts run without an install step."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

RAW = FIGURES / "raw"
DIAGRAM = FIGURES / "diagram"
RUNS = ROOT / "runs"
for _d in (RAW, DIAGRAM, RUNS):
    _d.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, kind: str = "diagram") -> Path:
    """Write a figure. ``kind`` is 'raw' or 'diagram' — see CONSTITUTION rule 8."""
    if kind not in ("raw", "diagram"):
        raise ValueError(f"figure kind must be 'raw' or 'diagram', got {kind!r}")
    path = (RAW if kind == "raw" else DIAGRAM) / name
    fig.savefig(path)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def run_dir(name: str) -> Path:
    """Directory for a long run's metrics and checkpoints — see CONSTITUTION 9."""
    path = RUNS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
