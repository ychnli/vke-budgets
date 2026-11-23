"""
scripts/check_merra2.py on a fake MERRA-2 tree: empty files for the name
check, tiny netCDF files for the DYN check.
"""
import importlib.util
from datetime import date
from pathlib import Path

import netCDF4
import pytest

from vke_budgets.sources import merra2

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_merra2.py"
spec = importlib.util.spec_from_file_location("check_merra2", SCRIPT)
check_merra2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_merra2)


def touch(root, collection, stamp, stream="100"):
    path = root / stamp[:4] / f"MERRA2_{stream}.{collection}.{stamp}.nc4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def full_day(root, stamp):
    for collection in check_merra2.LABELS:
        touch(root, collection, stamp)


def test_complete_days_need_the_next_inst(tmp_path):
    full_day(tmp_path, "19800101")
    full_day(tmp_path, "19800102")
    problems = check_merra2.check(str(tmp_path), date(1980, 1, 1), date(1980, 1, 2))
    assert problems[date(1980, 1, 1)] == []
    # The 2nd lacks the inst file of the 3rd, which closes its last window.
    assert problems[date(1980, 1, 2)] == ["inst 19800103 missing"]


def test_missing_streams_and_partial_files(tmp_path):
    full_day(tmp_path, "19801231")
    touch(tmp_path, "inst3_3d_asm_Np", "19810101")
    touch(tmp_path, "tavg3_3d_qdt_Np", "19801231", stream="101")      # a second stream
    (tmp_path / "1980" / "MERRA2_100.tavg3_3d_udt_Np.19801231.nc4").unlink()
    (tmp_path / "1980" / "MERRA2_100.tavg3_3d_udt_Np.19801231.nc4.123.part").touch()
    problems = check_merra2.check(str(tmp_path), date(1980, 12, 31), date(1980, 12, 31))
    assert sorted(problems[date(1980, 12, 31)]) == ["qdt 19801231 in 2 streams",
                                                     "udt 19801231 missing"]


def tendency_file(root, collection, stamp, with_dyn):
    path = root / stamp[:4] / f"MERRA2_100.{collection}.{stamp}.nc4"
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("time", 1)
        for name in merra2.VARIABLES[collection]:
            if with_dyn or not name.endswith(merra2.OPTIONAL_CODES):
                f.createVariable(name, "f4", ("time",))


@pytest.mark.parametrize("with_dyn", [True, False])
def test_dyn_check(tmp_path, with_dyn):
    for collection in ("inst3_3d_asm_Np", "tavg3_3d_asm_Nv"):
        touch(tmp_path, collection, "19900101")
    touch(tmp_path, "inst3_3d_asm_Np", "19900102")
    for collection in check_merra2.TENDENCIES:
        tendency_file(tmp_path, collection, "19900101", with_dyn)
    day = date(1990, 1, 1)
    assert check_merra2.check(str(tmp_path), day, day)[day] == []
    expected = [] if with_dyn else ["udt 19900101 without DYN", "qdt 19900101 without DYN"]
    assert check_merra2.check(str(tmp_path), day, day, with_dyn=True)[day] == expected


def test_summary_counts_months(tmp_path):
    for d in range(1, 32):
        full_day(tmp_path, f"198001{d:02d}")
    full_day(tmp_path, "19800201")
    problems = check_merra2.check(str(tmp_path), date(1980, 1, 1), date(1980, 2, 1))
    line = check_merra2.summary(problems)[1].split()
    assert line[:4] == ["1980", "32", "31", "1/2"]
