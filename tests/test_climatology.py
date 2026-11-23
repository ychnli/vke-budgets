"""
AR statistics by month (climatology.py) on synthetic input. A simple
threshold stands in for DetectBlobs, so these tests run without
TempestExtremes; detection itself is tested in test_detection.py.
"""
import os
from datetime import date

import numpy as np
import pytest
import xarray as xr

from synthetic import solid_body
from vke_budgets import budget, climatology, detection
from vke_budgets.sources import ibtracs

FAKE_TRACKS = xr.Dataset({"time": ("point", np.array(["2023-01-04"], "datetime64[ns]")),
                          "lat": ("point", [30.0]), "lon": ("point", [-140.0])},
                         attrs={"source": "fake tracks"})


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """Stand-ins for the IBTrACS file, so that run never downloads it."""
    monkeypatch.setattr(ibtracs, "ensure_tracks", lambda directory=None, log=None: "tracks.nc")
    monkeypatch.setattr(ibtracs, "open_tracks", lambda path=None, start=None, end=None: FAKE_TRACKS)


def fake_detect(columns, thresholds=None, tracks=None):
    """An "AR" wherever a column is above its global mean at that step."""
    fake_detect.tracks = tracks
    return xr.Dataset({f"mask_{f}": (columns[f] > columns[f].mean(("lat", "lon")))
                       .assign_attrs(long_name=f"{f} above its mean")
                       for f in climatology.FIELDS if f in columns})


def open_day(day, root):
    """Eight 3-hour windows starting at 00Z of `day`, like a MERRA-2 day."""
    return solid_body(n_time=8, start=day)


def test_months_days_and_tasks():
    months = climatology.months_between("2019-11", "2020-02")
    assert months == [date(2019, 11, 1), date(2019, 12, 1), date(2020, 1, 1), date(2020, 2, 1)]
    assert len(climatology.days_in_month(date(2020, 2, 1))) == 29
    assert climatology.days_in_month(date(2019, 12, 1))[-1] == "20191231"
    assert climatology.task_months(months, 0, 3) == months[:3]
    assert climatology.task_months(months, 1, 3) == months[3:]
    with pytest.raises(ValueError, match="tasks 0 to 1"):
        climatology.task_months(months, 2, 3)


def test_own_columns_leave_out_the_closing_instant():
    ds = open_day("20230104", "")
    columns = climatology.own_columns(ds)
    assert columns.sizes["time_inst"] == 8
    np.testing.assert_array_equal(columns["time_inst"], ds["time_inst"][:-1])
    expected = budget.instantaneous_columns(ds.isel(time_inst=slice(0, -1)))
    xr.testing.assert_allclose(columns, expected)


def test_ar_statistics():
    columns = climatology.own_columns(open_day("20230104", ""))
    masks = fake_detect(columns)
    stats = climatology.ar_statistics(columns, masks)
    assert int(stats["n_steps"]) == 8
    for f in climatology.FIELDS:
        np.testing.assert_array_equal(stats[f"ar_count_{f}"], masks[f"mask_{f}"].sum("time_inst"))
        np.testing.assert_allclose(stats[f"ar_frequency_{f}"], stats[f"ar_count_{f}"] / 8)
        np.testing.assert_allclose(stats[f"mean_{f}"], columns[f].mean("time_inst"))
        assert stats[f"ar_count_{f}"].dtype == np.int32


def test_run_writes_one_file_per_month_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(detection, "detect", fake_detect)
    out = str(tmp_path)
    log = []
    climatology.run("2023-01", "2023-02", "", out, save_steps=True, open_day=open_day,
                    log=log.append)
    for month, n_days in ((date(2023, 1, 1), 31), (date(2023, 2, 1), 28)):
        stats_path, steps_path = climatology.month_paths(out, month)
        with xr.open_dataset(stats_path) as stats, xr.open_dataset(steps_path) as steps:
            assert int(stats["n_steps"]) == 8 * n_days == steps.sizes["time_inst"]
            assert np.unique(steps["time_inst"]).size == steps.sizes["time_inst"]
            assert steps["mask_kev"].dtype == np.int8
            np.testing.assert_array_equal(steps["mask_kev"].sum("time_inst"),
                                          stats["ar_count_kev"])
            for name, value in detection.DEFAULT_THRESHOLDS.items():
                assert stats.attrs[f"threshold_laplacian_{name}"] == value
            assert stats.attrs["tc_exclusion"] == "4.0 degrees around fake tracks"
    assert fake_detect.tracks is FAKE_TRACKS
    assert not [f for f in os.listdir(out) if f.endswith(".tmp")]

    log.clear()
    climatology.run("2023-01", "2023-02", "", out, save_steps=True, open_day=open_day,
                    log=log.append)
    assert log == ["2023-01 exists, skipped", "2023-02 exists, skipped"]


def test_run_saves_masks_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(detection, "detect", fake_detect)
    out = str(tmp_path)
    climatology.run("2023-02", "2023-02", "", out, save_masks=True, open_day=open_day,
                    log=lambda message: None)
    assert sorted(os.listdir(out)) == ["ar_masks_202302.nc", "ar_stats_202302.nc"]
    with xr.open_dataset(climatology.masks_path(out, date(2023, 2, 1))) as masks, \
            xr.open_dataset(climatology.month_paths(out, date(2023, 2, 1))[0]) as stats:
        assert sorted(masks.data_vars) == ["mask_ivt", "mask_kev", "mask_vte"]
        assert masks.sizes["time_inst"] == 8 * 28 and masks["mask_kev"].dtype == np.int8
        np.testing.assert_array_equal(masks["mask_kev"].sum("time_inst"), stats["ar_count_kev"])
        assert masks.attrs["tc_exclusion"] == stats.attrs["tc_exclusion"]
    # A month with its statistics but without its masks is done again.
    os.remove(climatology.masks_path(out, date(2023, 2, 1)))
    log = []
    climatology.run("2023-02", "2023-02", "", out, save_masks=True, open_day=open_day,
                    log=log.append)
    assert "2023-02 exists, skipped" not in log
    assert os.path.exists(climatology.masks_path(out, date(2023, 2, 1)))


def test_run_one_task(tmp_path, monkeypatch):
    monkeypatch.setattr(detection, "detect", fake_detect)
    climatology.run("2023-01", "2023-03", "", str(tmp_path), task=1, months_per_task=2,
                    open_day=open_day, log=lambda message: None)
    assert sorted(os.listdir(tmp_path)) == ["ar_stats_202303.nc"]


def test_run_without_tc_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(detection, "detect", fake_detect)
    monkeypatch.setattr(ibtracs, "ensure_tracks", lambda *args, **kwargs: pytest.fail("download"))
    climatology.run("2023-01", "2023-01", "", str(tmp_path), exclude_tcs=False,
                    open_day=open_day, log=lambda message: None)
    assert fake_detect.tracks is None
    with xr.open_dataset(climatology.month_paths(str(tmp_path), date(2023, 1, 1))[0]) as stats:
        assert stats.attrs["tc_exclusion"] == "none"
