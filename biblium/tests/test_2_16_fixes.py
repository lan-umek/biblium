# -*- coding: utf-8 -*-
"""
Test za biblium 2.16 — preverja popravke iz OPTIMIZATION_REPORT.md.

POMEMBNO: Ta test najprej naloži biblium IZ TE MAPE (`biblium 2.16/`), in NE
iz globalno nameščenega paketa. Tako se zagotovo testirajo popravki tega
kodnega drevesa, ne katere koli prej nameščene starejše verzije.

Datoteka leži v `biblium 2.16/tests/`. Zagon (oba načina delujeta):

    pytest tests/test_2_16_fixes.py -v
    python tests/test_2_16_fixes.py
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

# `__file__` je v tests/, parent.parent je koren biblium-2.16 izvorne mape.
HERE = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loader: prisilno naloži biblium iz lokalnega drevesa (preglasi nameščeno).
# ---------------------------------------------------------------------------
def _load_biblium_from_local_tree():
    """Force-load biblium iz lokalne mape, ne iz `pip install`-ane verzije."""
    # 1) Pobriši vse cached `biblium*` module — sicer bi sys.modules še vedno
    #    kazal na staro nameščeno verzijo.
    for mod_name in list(sys.modules):
        if mod_name == "biblium" or mod_name.startswith("biblium."):
            del sys.modules[mod_name]

    init_path = HERE / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(
            f"Pričakoval sem __init__.py v {HERE}, ne najdem ga. "
            "Test mora ležati v korenu mape biblium 2.16."
        )

    # 2) Ročno registriraj paket pod imenom 'biblium' s submodule_search_locations
    #    vezanimi na to mapo. Ne glede na to, kako se mapa imenuje
    #    (`biblium 2.16` z presledkom in piko), bo Python od zdaj naprej
    #    `import biblium` in vse podmodule iskal v tej mapi.
    spec = importlib.util.spec_from_file_location(
        "biblium",
        init_path,
        submodule_search_locations=[str(HERE)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["biblium"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


# Naloži ob uvozu testnega modula. Če manjkajo težke odvisnosti (npr. pandas),
# bo to padlo tukaj — z razumljivo napako, in vsi testi bodo skipped.
try:
    biblium = _load_biblium_from_local_tree()
    LOAD_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 — namerno polovimo vse, da to ne podre testov
    biblium = None
    LOAD_ERROR = exc


def _skip_if_no_biblium():
    """Pomočnik: če biblium ni naložen, testu povemo, naj ga preskoči."""
    if biblium is None:
        try:
            import pytest
            pytest.skip(f"Biblium se ni naložil iz lokalnega drevesa: {LOAD_ERROR!r}")
        except ImportError:
            raise RuntimeError(
                f"Biblium se ni naložil iz lokalnega drevesa: {LOAD_ERROR!r}"
            )


# ---------------------------------------------------------------------------
# Sanity testi — preden testiramo popravke, preverimo, da imamo PRAVO verzijo.
# ---------------------------------------------------------------------------
def test_loaded_version_is_216():
    """Brez tega vsi naslednji testi ne pomenijo nič: če bi se naložila stara
    nameščena verzija, bi lahko slučajno »prešli« samo zato, ker so popravki v
    starih datotekah že drugje."""
    _skip_if_no_biblium()
    assert biblium.__version__ == "2.16.0", (
        f"Pričakoval sem biblium 2.16.0, naložila se je {biblium.__version__}. "
        "To pomeni, da test pomotoma uvaža staro nameščeno verzijo, ne lokalne."
    )


def test_loaded_path_points_to_this_folder():
    """Še bolj eksplicitno: __file__ paketa je v tej mapi."""
    _skip_if_no_biblium()
    pkg_file = Path(biblium.__file__).resolve()
    assert pkg_file.parent == HERE, (
        f"biblium se je naložil iz {pkg_file.parent}, "
        f"pričakovano pa je {HERE}."
    )


# ---------------------------------------------------------------------------
# Konkretni popravki F821 — »undefined name« bugov.
# ---------------------------------------------------------------------------
def test_dashboard_generator_imports_defaultdict():
    """Popravek: dashboard/generator.py je uporabljal `defaultdict(list)` brez
    importa `from collections import defaultdict`."""
    _skip_if_no_biblium()
    try:
        from biblium.dashboard import generator
    except ModuleNotFoundError as e:
        # Manjkajoča težka odvisnost (bokeh ipd.) — to ni naš popravek.
        try:
            import pytest
            pytest.skip(f"Manjkajoča odvisnost: {e}")
        except ImportError:
            return
    from collections import defaultdict
    assert hasattr(generator, "defaultdict"), (
        "defaultdict ni dostopen v dashboard.generator — manjkajoči import "
        "ni bil dodan."
    )
    assert generator.defaultdict is defaultdict


def test_utilsbib_modules_stats_has_Any():
    """Popravek: utilsbib_modules/stats.py je uporabljal `Any` v anotacijah
    brez `from typing import Any`."""
    _skip_if_no_biblium()
    try:
        from biblium.utilsbib_modules import stats
    except ModuleNotFoundError as e:
        try:
            import pytest
            pytest.skip(f"Manjkajoča odvisnost: {e}")
        except ImportError:
            return
    from typing import Any
    assert hasattr(stats, "Any"), "Any ni v utilsbib_modules.stats"
    assert stats.Any is Any


def test_correlation_module_loads_without_matplotlib_import():
    """Popravek: correlation.py je uporabljal `"matplotlib.figure.Figure"` kot
    forward reference, vendar matplotlib ni bil v `TYPE_CHECKING` bloku in
    je ruff javil F821. Modul se mora uvoziti tudi brez težav."""
    _skip_if_no_biblium()
    try:
        from biblium import correlation
    except ModuleNotFoundError as e:
        try:
            import pytest
            pytest.skip(f"Manjkajoča odvisnost: {e}")
        except ImportError:
            return
    assert hasattr(correlation, "compute_correlation")
    assert hasattr(correlation, "plot_correlation_matrix")


def test_main_path_module_loads():
    """Popravek: main_path.py — dodan TYPE_CHECKING blok za matplotlib."""
    _skip_if_no_biblium()
    try:
        from biblium import main_path
    except ModuleNotFoundError as e:
        try:
            import pytest
            pytest.skip(f"Manjkajoča odvisnost: {e}")
        except ImportError:
            return
    assert hasattr(main_path, "MainPathResult")


# ---------------------------------------------------------------------------
# Statična preverjanja — kar smo ODSTRANILI iz datotek, naj res ne bo več.
# ---------------------------------------------------------------------------
def test_addons_dead_axes_pattern_removed():
    """Popravek: odstranjen mrtvi `for _ax in axes.flat: _` blok v 7
    funkcijah v addonih (axes ni bil definiran, _ je useless expression)."""
    addons = [
        "addons/comparative_analysis.py",
        "addons/dynamic_topic_models.py",
        "addons/impact_metrics.py",
        "addons/temporal_networks.py",
    ]
    for rel in addons:
        path = HERE / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "for _ax in axes.flat:\n        _\n" not in text, (
            f"{rel} še vedno vsebuje mrtvi vzorec `for _ax in axes.flat: _`"
        )


def test_lambda_e_bug_pattern_removed_in_panels():
    """Popravek: `lambda: ...str(e)` znotraj `except Exception as e:` je bug —
    `e` se izbriše ob izhodu iz except bloka, lambda pa se izvede kasneje
    preko `self.after(0, …)`. Pravilen vzorec je `lambda msg=str(e): …`."""
    bad = re.compile(r"lambda:\s*self\._on_\w+_error\(str\(e\)\)")
    panel_files = [
        "gui/panels/analysis/overview_panel.py",
        "gui/panels/analysis/trends_panel.py",
        "gui/panels/analysis/sdg_panel.py",
        "gui/panels/data/filter_panel.py",
        "gui/panels/analysis/reference_diversity_panel.py",
        "gui/panels/analysis/top_cited_panel.py",
    ]
    for rel in panel_files:
        path = HERE / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        matches = bad.findall(text)
        assert not matches, (
            f"{rel} še vedno vsebuje `lambda: …self._on_*_error(str(e))` "
            f"vzorec: {matches}"
        )


def test_concept_builder_dead_code_removed():
    """Popravek: `_export_filtered_data()` v concept_builder_panel.py je
    vseboval mrtvo kodo `self._current_result = summary_df`, kjer summary_df
    NI parametriziran v tej funkciji (kopiran iz `_on_success`)."""
    path = HERE / "gui/panels/analysis/concept_builder_panel.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "_export_filtered_data" not in text:
        return
    start = text.find("def _export_filtered_data")
    rest = text[start:]
    next_def = re.search(r"\n    def ", rest[1:])
    end = next_def.start() + 1 if next_def else len(rest)
    block = rest[:end]
    assert "self._current_result = summary_df" not in block, (
        "Mrtva koda `self._current_result = summary_df` je še vedno "
        "v _export_filtered_data — undefined `summary_df` bo sprožil NameError."
    )


def test_no_duplicated_ax_grid_calls():
    """Popravek: 72 podvojenih zaporednih `ax.grid(False)` klicev odstranjenih.
    V naključnem vzorcu addonov ne sme biti DVA zaporedna identična klica."""
    pattern = re.compile(r"^    ax\.grid\(False\)\n    ax\.grid\(False\)$",
                         re.MULTILINE)
    sampled = [
        "addons/altmetrics.py",
        "addons/comparative_analysis.py",
        "addons/text_mining_nlp.py",
    ]
    for rel in sampled:
        path = HERE / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        m = pattern.search(text)
        assert m is None, (
            f"{rel} ima še vedno podvojen `ax.grid(False)` blok"
        )


# ---------------------------------------------------------------------------
# Funkcionalni: lambda-e popravek delovno preverimo (brez Tkinterja).
# ---------------------------------------------------------------------------
def test_lambda_default_arg_pattern_actually_captures_value():
    """Pozitivno preverjanje, da nov vzorec `lambda msg=str(e): …` res ohrani
    napako tudi po izhodu iz except bloka. To je natanko bug, ki smo ga
    popravili v ~26 lokacijah, in to je njegov idiomatski test."""
    captured = []

    def producer():
        try:
            raise ValueError("test sporočilo")
        except ValueError as e:
            # Točno popravek, ki smo ga uporabili:
            return lambda msg=str(e): captured.append(msg)
        # Tu se 'e' izbriše. Lambda je že shranjena s privzetim argumentom.

    callback = producer()
    callback()  # izvedba kasneje, izven except bloka
    assert captured == ["test sporočilo"], (
        f"Privzeti-argument vzorec ni ujel vrednosti: {captured!r}"
    )


# ---------------------------------------------------------------------------
# CLI runner — `python test_biblium_216_fixes.py` (brez pytest).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    tests = [
        ("Naložena verzija = 2.16.0", test_loaded_version_is_216),
        ("biblium se naloži iz te mape", test_loaded_path_points_to_this_folder),
        ("dashboard.generator: defaultdict importan", test_dashboard_generator_imports_defaultdict),
        ("utilsbib_modules.stats: Any importan", test_utilsbib_modules_stats_has_Any),
        ("correlation.py: uvoz brez napak", test_correlation_module_loads_without_matplotlib_import),
        ("main_path.py: uvoz brez napak", test_main_path_module_loads),
        ("addons: mrtvi axes-pattern odstranjen", test_addons_dead_axes_pattern_removed),
        ("GUI paneli: lambda-e bug ne pojavlja več", test_lambda_e_bug_pattern_removed_in_panels),
        ("concept_builder: mrtva koda v export", test_concept_builder_dead_code_removed),
        ("addons: ni več podvojenih ax.grid()", test_no_duplicated_ax_grid_calls),
        ("lambda(msg=str(e)) idiom delovno preverjen", test_lambda_default_arg_pattern_actually_captures_value),
    ]

    print(f"\n=== Test biblium 2.16 (lokalno: {HERE}) ===\n")
    if biblium is not None:
        print(f"Naložena verzija: {biblium.__version__}")
    else:
        print(f"OPOZORILO: biblium se ni naložil — {LOAD_ERROR!r}")
    print()

    # `pytest.skip` vrže pytest.outcomes.Skipped — to ujamemo kot SKIP, ne FAIL.
    try:
        from _pytest.outcomes import Skipped as _Skipped
    except ImportError:
        class _Skipped(Exception):  # type: ignore
            pass

    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK    {name}")
            passed += 1
        except _Skipped as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            print("        " + tb.replace("\n", "\n        "))
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(0 if failed == 0 else 1)
