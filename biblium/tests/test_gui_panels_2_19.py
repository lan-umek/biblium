# -*- coding: utf-8 -*-
"""
GUI smoke tests (2.19): the ``biblium.gui`` package must import, the app must
construct, and every registered panel must instantiate without errors.

Runs only where a Tk display is available (Windows/macOS desktop, or Linux
with DISPLAY/Xvfb); otherwise the tests are skipped. In 2.16/2.18 the GUI
package could not even be imported (``gui.panels.reports`` was missing), which
this test now guards against.
"""
from __future__ import annotations

import importlib

import pytest

tk = pytest.importorskip("tkinter")
try:  # needs a real Tk build (the TkAgg backend imports _tkinter)
    import matplotlib.backends.backend_tkagg  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    pytest.skip(f"TkAgg backend unavailable: {_exc}", allow_module_level=True)


def _tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception as exc:  # noqa: BLE001 - no display
        pytest.skip(f"No Tk display available: {exc}")


def test_gui_package_imports():
    mod = importlib.import_module("biblium.gui.panels")
    for name in ("BasePanel", "ReportBuilderPanel", "CustomReportPanel", "TopicsPanel"):
        assert hasattr(mod, name), name
    v219 = importlib.import_module("biblium.gui.panels.v219")
    assert len(v219.V219_PANELS) >= 20


def test_app_constructs_and_all_panels_instantiate():
    root = _tk_root()
    try:
        from biblium.gui.app import BibliumApp
        app = BibliumApp()
        try:
            registry = app.workspace._panel_registry
            assert len(registry) >= 90, len(registry)
            failures = []
            for pid, cls in registry.items():
                try:
                    panel = cls(app.workspace, bib=None)
                    panel.destroy()
                except Exception as exc:  # noqa: BLE001
                    failures.append((pid, repr(exc)[:200]))
            assert not failures, failures
        finally:
            try:
                app.destroy()
            except Exception:
                pass
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_simple_panel_round_trip():
    """A SimpleAnalysisPanel subclass runs compute() and renders tables/plots."""
    root = _tk_root()
    try:
        import pandas as pd
        from biblium.gui.panels.simple import SimpleAnalysisPanel, Opt

        class Demo(SimpleAnalysisPanel):
            OPTIONS = [Opt("n", "N", "spin", 3, from_=1, to=9), Opt("flag", "Flag", "check", True)]

            def compute(self, bib, o):
                return {"summary": [("n", o["n"])], "tables": {"T": pd.DataFrame({"a": [o["n"]]})},
                        "plots": {"P": lambda fig: fig.add_subplot(111).plot([1, 2])}}

        class FakeBib:
            df = pd.DataFrame({"Year": [2000]})

        panel = Demo(root, bib=FakeBib())
        assert panel.get_options()["n"] == 3
        panel._render(panel.compute(panel.bib, panel.get_options()))
        assert "T" in panel._tables
        panel.destroy()
    finally:
        root.destroy()
