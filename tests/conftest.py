import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REFERENCE_PY = REPO / "reference" / "py"
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def tools_comp():
    """The reference operators, reference/py/tools_comp.py, from the gitignored
    snapshot in reference/. Tests that need it are skipped without it."""
    if not REFERENCE_PY.is_dir():
        pytest.skip("reference/ snapshot not present")
    pytest.importorskip("metpy")
    sys.path.insert(0, str(REFERENCE_PY))
    import tools_comp
    return tools_comp
