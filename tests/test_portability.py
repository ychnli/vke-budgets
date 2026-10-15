"""No cluster-specific paths or settings in the files that are distributed.
Site details belong in the local, gitignored scripts/env.sh (copied from
scripts/env.example.sh) and in environment variables."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKED = ["src", "scripts", "tests", "README.md", "pyproject.toml"]
LOCAL = {REPO / "scripts" / "env.sh", Path(__file__).resolve()}
SITE_SPECIFIC = re.compile(r"/home/|/scratch/|/oak/|GROUP_SCRATCH|GROUP_HOME|\$SCRATCH|"
                           r"--account=\w|--partition=\w|Aidi")


def checked_files():
    for name in CHECKED:
        path = REPO / name
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            if f not in LOCAL and "__pycache__" not in f.parts and f.suffix != ".pyc" \
                    and ".egg-info" not in str(f):
                yield f


def test_no_site_specific_paths():
    hits = []
    for f in checked_files():
        for number, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if SITE_SPECIFIC.search(line):
                hits.append(f"{f.relative_to(REPO)}:{number}: {line.strip()}")
    assert not hits, "site-specific lines:\n" + "\n".join(hits)
