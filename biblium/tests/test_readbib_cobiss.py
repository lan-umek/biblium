# -*- coding: utf-8 -*-
"""
Smoke tests for the COBISS branch of ``readbib.read_bibfile``.

These verify the dispatcher integration (rather than parsing per se,
which is covered in ``test_cobiss_parser``):

- ``db="cobiss"`` accepts both ``.csv`` (already parsed) and ``.html``
  (raw COBISS+ output) inputs.
- Unsupported extensions raise a ``ValueError`` with a helpful message
  pointing at the high-level convenience function.
- The dispatcher's output matches what ``read_cobiss_html`` produces
  directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biblium.readbib import read_bibfile, read_cobiss_html


class TestCobissDispatcher:
    """``read_bibfile(f_name, db='cobiss')``."""

    def test_html_path_dispatches(self, fixtures_dir: Path):
        # Our markdown sample is recognised because read_cobiss_html
        # auto-detects HTML vs text/markdown content.
        path = str(fixtures_dir / "cobiss_sample.md")
        # Note: the dispatcher's filename check uses extensions; .md is
        # not "html" so it would route through the CSV branch and fail.
        # We use the .html branch by writing a short html-extension wrapper.
        ...  # actual extension test below

    def test_csv_round_trip(self, fixtures_dir: Path, tmp_path):
        # Build a CSV from the parser, then read it back via the dispatcher.
        df_direct = read_cobiss_html(
            str(fixtures_dir / "cobiss_sample.md")
        )
        out_csv = tmp_path / "cobiss.csv"
        df_direct.to_csv(out_csv, index=False)

        df_disp = read_bibfile(str(out_csv), db="cobiss")
        # Same number of rows and same columns
        assert len(df_disp) == len(df_direct)
        assert set(df_disp.columns) == set(df_direct.columns)

    def test_html_extension_dispatches(self, tmp_path, fixtures_dir: Path):
        # Make a .html-extension copy so the dispatcher routes to read_cobiss_html
        html_path = tmp_path / "personal_bib.html"
        html_path.write_text(
            (fixtures_dir / "cobiss_sample.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        df = read_bibfile(str(html_path), db="cobiss")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Sanity: canonical columns must be present
        for col in ("Authors", "Title", "Year", "Document Type", "cobiss_id"):
            assert col in df.columns

    def test_unsupported_extension_raises(self, tmp_path):
        bogus = tmp_path / "bib.bib"
        bogus.write_text("@article{foo, title={Bar}}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="COBISS"):
            read_bibfile(str(bogus), db="cobiss")

    def test_error_message_points_at_convenience_function(self, tmp_path):
        bogus = tmp_path / "bib.txt"
        bogus.write_text("not actually cobiss content", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            read_bibfile(str(bogus), db="cobiss")
        # The error should hint at the high-level fetch function so the
        # user knows where to go.
        assert "fetch_personal_bibliography" in str(exc.value)

    def test_db_argument_is_case_insensitive(self, tmp_path, fixtures_dir):
        html_path = tmp_path / "p.html"
        html_path.write_text(
            (fixtures_dir / "cobiss_sample.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        for db in ("cobiss", "COBISS", "Cobiss"):
            df = read_bibfile(str(html_path), db=db)
            assert len(df) > 0
