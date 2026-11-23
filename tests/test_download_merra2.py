"""
The skip decision of scripts/download_merra2.py (up_to_date) on tiny netCDF
files; nothing is downloaded.
"""
import importlib.util
from pathlib import Path

import netCDF4
import pytest

from vke_budgets.sources import merra2

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_merra2.py"
spec = importlib.util.spec_from_file_location("download_merra2", SCRIPT)
download_merra2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_merra2)

UDT = "tavg3_3d_udt_Np"


def tendency_file(path, with_dyn):
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("time", 1)
        for name in download_merra2.variables(UDT, with_dyn):
            if name not in f.variables:
                f.createVariable(name, "f4", ("time",))
    return str(path)


@pytest.mark.parametrize("has_dyn", [True, False])
def test_without_with_dyn_any_existing_file_is_kept(tmp_path, has_dyn):
    path = tendency_file(tmp_path / "f.nc4", has_dyn)
    assert download_merra2.up_to_date(path, download_merra2.variables(UDT, with_dyn=False))


@pytest.mark.parametrize("has_dyn", [True, False])
def test_with_dyn_a_file_without_dyn_is_downloaded_again(tmp_path, has_dyn):
    path = tendency_file(tmp_path / "f.nc4", has_dyn)
    assert download_merra2.up_to_date(path, download_merra2.variables(UDT, with_dyn=True)) \
        == has_dyn


def test_missing_or_unreadable_files_are_downloaded(tmp_path):
    names = download_merra2.variables(UDT, with_dyn=True)
    assert not download_merra2.up_to_date(str(tmp_path / "absent.nc4"), names)
    broken = tmp_path / "broken.nc4"
    broken.write_bytes(b"not a netCDF file")
    assert not download_merra2.up_to_date(str(broken), names)


def test_collections_without_dyn_are_never_opened(tmp_path):
    """inst and Nv have no DYN variable, so --with-dyn doesn't open them:
    even an unreadable file counts as up to date."""
    broken = tmp_path / "inst.nc4"
    broken.write_bytes(b"not a netCDF file")
    for collection in ("inst3_3d_asm_Np", "tavg3_3d_asm_Nv"):
        assert download_merra2.up_to_date(str(broken),
                                          download_merra2.variables(collection, with_dyn=True))
    assert merra2.OPTIONAL_CODES == ("DYN",)
