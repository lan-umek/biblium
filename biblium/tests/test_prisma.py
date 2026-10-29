# -*- coding: utf-8 -*-
"""Tests for ``biblium.prisma`` (new in 2.19): derived arithmetic, table
shape and rendering of the PRISMA 2020 diagram (hermetic, Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from biblium import prisma as P


class TestFillDerived:
    def test_forward_chain(self):
        c = P.PrismaCounts(db_records=1200, register_records=30, duplicates_removed=200,
                           records_excluded=700, reports_not_retrieved=10,
                           exclusion_reasons={"A": 40, "B": 25, "C": 5})
        c.fill_derived()
        # 1230 − 200 = 1030; 1030 − 700 = 330; 330 − 10 = 320; 320 − 70 = 250
        assert c.records_screened == 1030
        assert c.reports_sought == 330
        assert c.reports_assessed == 320
        assert c.reports_excluded == 70
        assert c.studies_included == 250
        assert c.reports_included == 250
        assert c.validate() == []

    def test_backward_solving(self):
        # screened and sought known → excluded derived; assessed and included → reports excluded
        c = P.PrismaCounts(records_screened=500, reports_sought=120, reports_assessed=110,
                           studies_included=40).fill_derived()
        assert c.records_excluded == 380
        assert c.reports_not_retrieved == 10
        assert c.reports_excluded == 70

    def test_missing_removal_term_solved(self):
        c = P.PrismaCounts(db_records=100, records_screened=80, removed_automation=5,
                           removed_other=0).fill_derived()
        assert c.duplicates_removed == 15
        # two unknown removal terms cannot be solved
        c2 = P.PrismaCounts(db_records=100, records_screened=80).fill_derived()
        assert c2.duplicates_removed is None

    def test_existing_values_untouched_and_validate_flags_inconsistency(self):
        c = P.PrismaCounts(db_records=100, duplicates_removed=10, records_screened=95).fill_derived()
        assert c.records_screened == 95
        assert c.validate() == ["identified − removed ≠ records_screened"]

    def test_nothing_derivable(self):
        c = P.PrismaCounts(db_records=10).fill_derived()
        # no removals known → screened = 10 − 0
        assert c.records_screened == 10
        assert c.reports_sought is None


class TestPipelineHelper:
    def test_with_screening_stage(self):
        c = P.prisma_counts_from_pipeline(1500, 1300, 120, n_screened_out=1000,
                                          exclusion_reasons={"A": 100, "B": 80})
        assert c.duplicates_removed == 200
        assert c.records_excluded == 1000
        assert c.reports_sought == 300 and c.reports_assessed == 300
        assert c.reports_excluded == 180 and c.studies_included == 120
        assert c.validate() == []

    def test_without_screening_stage(self):
        c = P.prisma_counts_from_pipeline(1500, 1300, 120)
        assert c.records_excluded == 1180
        assert c.reports_sought == 120 and c.reports_excluded == 0
        assert c.validate() == []

    def test_extra_sources_and_removals(self):
        c = P.prisma_counts_from_pipeline(1000, 800, 50, n_register_records=100,
                                          n_removed_automation=20, n_not_retrieved=3)
        assert c.duplicates_removed == 1100 - 800 - 20
        assert c.reports_sought == 53 and c.reports_assessed == 50


class TestDataFrame:
    def test_shape_and_reasons(self):
        c = P.PrismaCounts(db_records=10, exclusion_reasons={"x": 1, "y": 2})
        df = c.to_dataframe()
        assert list(df.columns) == ["Stage", "Count"]
        assert len(df) == len(P.PRISMA_STAGES) + 2
        assert df.loc[df["Stage"] == "Records identified from databases", "Count"].iloc[0] == 10
        assert df.loc[df["Stage"] == "Reports excluded: y", "Count"].iloc[0] == 2
        assert pd.isna(df.loc[df["Stage"] == "Records screened", "Count"].iloc[0])

    def test_as_dict_roundtrip(self):
        c = P.PrismaCounts(db_records=5, labels={"screened": "X"})
        d = c.as_dict()
        assert "labels" not in d and d["db_records"] == 5 and d["exclusion_reasons"] is None


class TestPlot:
    def test_render_and_save(self, tmp_path):
        c = P.PrismaCounts(db_records=1200, register_records=30, other_records=5,
                           duplicates_removed=200, removed_automation=3, records_excluded=700,
                           reports_not_retrieved=10,
                           exclusion_reasons={"Wrong population": 40, "Not empirical": 25},
                           labels={"screening": "Presejanje"})
        out = tmp_path / "prisma.png"
        fig, ax = P.plot_prisma(c, title="PRISMA", f_name=str(out), dpi=60)
        assert out.exists() and out.stat().st_size > 0
        assert ax.get_title() == "PRISMA"
        texts = [t.get_text() for t in ax.texts]
        assert "Presejanje" in texts
        # derived records screened: 1200 + 30 + 5 − 200 − 3 = 1032
        assert any("1,032" in t for t in texts)
        assert any("Wrong population" in t for t in texts)
        # input must not be mutated by plotting
        assert c.records_screened is None
        matplotlib.pyplot.close(fig)

    def test_boxes_do_not_overlap(self):
        from matplotlib.patches import FancyBboxPatch
        c = P.prisma_counts_from_pipeline(1500, 1300, 120, n_screened_out=1000,
                                          exclusion_reasons={f"Reason {i}": i for i in range(1, 7)})
        fig, ax = P.plot_prisma(c)
        boxes = [p.get_bbox() for p in ax.patches if isinstance(p, FancyBboxPatch)]
        assert len(boxes) == 9 + 3  # 9 content boxes + 3 phase labels
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                overlap = (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)
                assert not overlap, (i, j)
        matplotlib.pyplot.close(fig)

    def test_existing_axis(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig2, ax2 = P.plot_prisma(P.PrismaCounts(db_records=3), ax=ax)
        assert ax2 is ax and fig2 is fig
        plt.close(fig)
