"""Writing, reading and the multi-day pipeline, on synthetic input."""
import os

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from synthetic import solid_body
from vke_budgets import budget, io, pipeline


def day_result(day="2023-01-04", keep_3d=False):
    ds = solid_body(n_time=8, drag=1e-5, drying=1e-8, start=day)
    return budget.compute(ds, budget.BudgetOptions(keep_3d=keep_3d))


def test_result_spans_its_bounding_steps():
    result = day_result()
    assert result.sizes["time"] == 8 and result.sizes["time_inst"] == 9


def test_clean_round_trip_drops_the_closing_step(tmp_path):
    result = day_result()
    path = str(tmp_path / "day.nc")
    io.write_period(result, path)
    back = io.open_budget(path).load()
    assert back.sizes["time_inst"] == 8
    xr.testing.assert_allclose(back.drop_dims("time_inst"),
                               result.drop_dims("time_inst").astype("float32"))
    xr.testing.assert_allclose(back["IKEV"], result["IKEV"].isel(time_inst=slice(0, 8)).astype("float32"))


def test_consecutive_files_open_as_one(tmp_path):
    paths = []
    for day in ("2023-01-04", "2023-01-05"):
        paths.append(str(tmp_path / f"{day}.nc"))
        io.write_period(day_result(day), paths[-1])
    both = io.open_budget(str(tmp_path / "*.nc"))
    assert both.sizes == {"time": 16, "time_inst": 16, "lat": both.sizes["lat"],
                          "lon": both.sizes["lon"]}
    assert pd.DatetimeIndex(both["time_inst"].values).is_monotonic_increasing


def test_bitround_precision(tmp_path):
    result = day_result()
    path = str(tmp_path / "rounded.nc")
    io.write_period(result, path, significant_bits=12)
    back = io.open_budget(path).load()
    a, b = back["ITOKE"].values, result["ITOKE"].values
    ok = np.isfinite(b)
    assert 0 < np.abs(a[ok] / b[ok] - 1).max() < 2.0**-12


def test_legacy_layout_round_trip(tmp_path):
    result = day_result(keep_3d=True)
    io.write_legacy(result, str(tmp_path), "20230104")
    inst_path, tavg_path = io.legacy_paths(str(tmp_path), "20230104")
    with netCDF4.Dataset(tavg_path) as f:
        assert f["time"].units == "minutes since 2023-01-04 01:30:00"
        assert list(f["time"][:]) == list(range(0, 8 * 180, 180))
        assert f["IHAKE"]._FillValue == np.float32(1e15)
        assert f["IHAKE"].units == "kg / s^3"
        assert f["lev"].units == "hPa" and f["lev"][0] == 1000.0
    with netCDF4.Dataset(inst_path) as f:
        assert f["time"].units == "minutes since 2023-01-04 00:00:00"
        assert f.dimensions["time"].size == 8
    back = io.open_legacy(str(tmp_path), "20230104")
    xr.testing.assert_allclose(back["IHAKE"], result["IHAKE"].astype("float32"))
    xr.testing.assert_allclose(back["HAKE"], result["HAKE"].astype("float32"))
    xr.testing.assert_allclose(back["IVT"], result["IVT"].isel(time_inst=slice(0, 8)).astype("float32"))


# The reference's 2017-07-12 outputs (analysis_AR_IVKE1_{inst,tavg}20170712.nc).
REF_2017 = os.environ.get("VKE_TEST_REFERENCE", "")


@pytest.mark.skipif(not os.path.isdir(REF_2017), reason="set VKE_TEST_REFERENCE to run")
def test_open_legacy_reads_the_reference_files():
    ref = io.open_legacy(REF_2017, "20170712")
    assert ref.sizes["time"] == 8 and ref.sizes["time_inst"] == 8
    assert ref["IHAKE"].dims == ("time", "lat", "lon")
    assert str(ref["time"].values[0])[:16] == "2017-07-12T01:30"
    assert np.isnan(ref["IHAKE"].values[0, 0]).all()        # pole rows blanked, fill decoded


def fake_source(calls):
    """open_day for the pipeline: synthetic input, counting the days opened."""
    def open_day(date, root):
        calls.append(date)
        return solid_body(n_time=8, drag=1e-5, start=pd.Timestamp(date))
    return open_day


def test_pipeline_months_and_resume(tmp_path):
    out, calls = str(tmp_path / "out"), []
    pipeline.run("2023-01-30", "2023-02-01", None, out, open_day=fake_source(calls), log=None)
    files = sorted(os.listdir(out))
    assert files == ["budget_20230130-20230131.nc", "budget_20230201-20230201.nc"]
    both = io.open_budget(os.path.join(out, "budget_*.nc"))
    assert both.sizes["time"] == 24 and both.sizes["time_inst"] == 24
    assert calls == ["20230130", "20230131", "20230201"]

    # Rerun: complete months are skipped.
    pipeline.run("2023-01-30", "2023-02-01", None, out, open_day=fake_source(calls), log=None)
    assert len(calls) == 3

    # A month interrupted after one day resumes from its part file.
    os.remove(os.path.join(out, "budget_20230130-20230131.nc"))
    os.makedirs(os.path.join(out, "parts"))
    io.write_period(budget.compute(solid_body(n_time=8, start="2023-01-30")),
                    os.path.join(out, "parts", "20230130.nc"))
    pipeline.run("2023-01-30", "2023-01-31", None, out, open_day=fake_source(calls), log=None)
    assert calls[3:] == ["20230131"]
    assert not os.path.exists(os.path.join(out, "parts"))


def test_pipeline_legacy(tmp_path):
    calls = []
    pipeline.run("2023-01-04", "2023-01-05", None, str(tmp_path), layout="legacy",
                 open_day=fake_source(calls), log=None)
    assert sorted(os.listdir(tmp_path)) == [
        "analysis_AR_IVKE1_inst20230104.nc", "analysis_AR_IVKE1_inst20230105.nc",
        "analysis_AR_IVKE1_tavg20230104.nc", "analysis_AR_IVKE1_tavg20230105.nc"]
