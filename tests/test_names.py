"""The name table: complete, reversible, and documented."""
from pathlib import Path

from vke_budgets import names

REPO = Path(__file__).resolve().parent.parent


def test_legacy_translation_is_one_to_one():
    table = names.legacy_names()
    assert len(set(table.values())) == len(table)


def test_every_group_has_members():
    for group in names.GROUPS:
        assert names.members(group), group


def test_reference_names():
    table = names.legacy_names()
    assert table["kev_hflux_ke"] == "IHAKE" and table["vte_hflux_ke"] == "IHAKE_v"
    assert table["kev_hflux_ke_3d"] == "HAKE"
    assert table["kev_tend"] == "IKEVTEND" and table["vte_tend_local"] == "I_VTETEND"
    assert table["kev_boundary"] == "SPTE" and table["vte_boundary"] == "SPTE_v"
    assert table["vte_hflux_vapor_ncl"] == "IHCV_vO"
    assert table["kev_mean_3d"] == "KEV" and table["ivt"] == "IVT"


def test_docs_table_is_current():
    """docs/names.md is generated from the table; regenerate it with
    python -m vke_budgets.names > docs/names.md"""
    assert (REPO / "docs" / "names.md").read_text() == names.translation_table()
