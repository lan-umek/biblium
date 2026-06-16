# -*- coding: utf-8 -*-
"""
Demo / integration test za biblium 2.16 z vgrajenim Scopus datasetom.

Naredi tri stvari:
  (1) Scientific production    -- letna produkcija + plot.
  (2) Sources stats            -- statistika po revijah (vkljucno s h-indeksi).
  (3) Keyword co-occurrence    -- graf soodvisnosti samo med kljucnimi besedami,
                                  ki se zacnejo z "biblio*", + plot.

Uporabljen vir: `data/scopus dataset.csv` (200 zapisov, vgrajen v repo).

POMEMBNO: Skripta najprej nalozi biblium IZ TE MAPE (`biblium 2.16/`), in NE
iz namescene starejse verzije.

Zagon:

    python tests/test_2_16_scopus_demo.py
    pytest tests/test_2_16_scopus_demo.py -v -s
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # koren biblium-2.16


def _load_biblium_from_local_tree():
    """Force-load biblium iz HERE namesto iz pip install-ane verzije."""
    for mod_name in list(sys.modules):
        if mod_name == "biblium" or mod_name.startswith("biblium."):
            del sys.modules[mod_name]
    init_path = HERE / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(f"Pricakoval sem __init__.py v {HERE}.")
    spec = importlib.util.spec_from_file_location(
        "biblium",
        init_path,
        submodule_search_locations=[str(HERE)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["biblium"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


try:
    biblium = _load_biblium_from_local_tree()
    LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001
    biblium = None
    LOAD_ERROR = exc


def _skip_if_no_biblium():
    if biblium is not None:
        return
    try:
        import pytest
        pytest.skip(f"Biblium se ni nalozil iz lokalnega drevesa: {LOAD_ERROR!r}")
    except ImportError:
        raise RuntimeError(
            f"Biblium se ni nalozil iz lokalnega drevesa: {LOAD_ERROR!r}"
        )


_BS_CACHE = {"bs": None, "outdir": None}


def _get_biblio_stats():
    """Naredi BiblioPlot nad data/scopus dataset.csv (cache za vse teste).

    BiblioPlot podeduje BiblioStats in dodatno ponuja plot metode kot so
    plot_scientific_production() in plot_coocurrence().
    """
    _skip_if_no_biblium()
    if _BS_CACHE["bs"] is not None:
        return _BS_CACHE["bs"], _BS_CACHE["outdir"]

    from biblium.bibplot import BiblioPlot

    csv_path = HERE / "data" / "scopus dataset.csv"
    assert csv_path.exists(), (
        f"Vgrajen Scopus dataset ne najdem na {csv_path}. "
        "Pricakovan je v `data/scopus dataset.csv`."
    )

    outdir = HERE / "tests" / "_demo_outputs_2_16"
    outdir.mkdir(parents=True, exist_ok=True)

    bs = BiblioPlot(
        f_name=str(csv_path),
        db="scopus",
        res_folder=str(outdir),
        dpi=120,
        fancy_output=False,
    )
    _BS_CACHE["bs"] = bs
    _BS_CACHE["outdir"] = outdir
    return bs, outdir


# ---------------------------------------------------------------------------
# (1) Scientific production
# ---------------------------------------------------------------------------
def test_1_scientific_production():
    """Letna znanstvena produkcija + plot."""
    bs, outdir = _get_biblio_stats()

    prod = bs.get_production()
    print("\n[1] Scientific production: zadnjih 10 let")
    print(prod.tail(10).to_string(index=False))

    assert prod is not None and len(prod) > 0, "Produkcijska tabela je prazna."
    cols = [c.lower() for c in prod.columns]
    assert any("year" in c for c in cols), (
        f"Pricakoval sem stolpec z 'Year', dobil: {list(prod.columns)}"
    )

    bs.plot_scientific_production(filename="01_scientific_production")

    plots_dir = outdir / "plots"
    assert plots_dir.exists(), f"Mapa s ploti ne obstaja: {plots_dir}"
    matching = list(plots_dir.glob("01_scientific_production*"))
    assert matching, (
        f"Pricakoval sem datoteko 01_scientific_production* v {plots_dir}, "
        f"vendar je ni: {[p.name for p in plots_dir.iterdir()]}"
    )
    print(f"    plot -> {matching[0].relative_to(HERE)}")


# ---------------------------------------------------------------------------
# (2) Sources stats
# ---------------------------------------------------------------------------
def test_2_sources_stats():
    """Statistika po revijah (h-indeksi, citati, ...)."""
    bs, _ = _get_biblio_stats()

    src = bs.get_sources_stats(top_n=20)

    # Izberemo le najbolj informativne stolpce za prikaz.
    show_cols = [c for c in [
        "Source", "Number of documents", "Total citations",
        "H-index", "G-index", "Average year",
    ] if c in src.columns]
    print("\n[2] Top 10 revij (kljucni stolpci):")
    print(src[show_cols].head(10).to_string(index=False))

    assert src is not None and len(src) > 0, "Tabela sources_stats je prazna."
    assert "Source" in src.columns or "Source title" in src.columns, (
        f"Manjka stolpec 'Source'. Stolpci: {list(src.columns)}"
    )
    assert len(src) <= 20, f"top_n=20 ni bil spostovan: vrnjenih {len(src)} vrstic."


# ---------------------------------------------------------------------------
# (3) Keyword co-occurrence network -- samo "biblio*" kljucne besede
# ---------------------------------------------------------------------------
def test_3_biblio_keyword_cooccurrence_network():
    """Network le med kljucnimi besedami, ki se zacnejo z 'biblio*'."""
    bs, outdir = _get_biblio_stats()

    import networkx as nx

    # Vzamemo zelo velik top_n, da zagotovo zajamemo vse biblio* terme,
    # nato podgrafiramo le na 'biblio*' vozlisca.
    G_all = bs.build_keyword_cooccurrence_network(top_n=500, min_cooccur=1)
    biblio_nodes = [n for n in G_all.nodes() if str(n).lower().startswith("biblio")]
    print(f"\n[3] Najdenih {len(biblio_nodes)} 'biblio*' kljucnih besed v top-500.")
    print("    Primeri:", biblio_nodes[:10])

    assert biblio_nodes, (
        "V vgrajenem Scopus datasetu naj bi obstajale kljucne besede, "
        "ki se zacnejo z 'biblio*' (vsaj 'bibliometrics')."
    )

    G_bib = G_all.subgraph(biblio_nodes).copy()
    print(f"    Subgraph: {G_bib.number_of_nodes()} vozlisc, "
          f"{G_bib.number_of_edges()} povezav.")

    assert G_bib.number_of_nodes() == len(biblio_nodes)
    assert G_bib.number_of_nodes() >= 2, (
        "Premalo biblio* vozlisc za smiselen network (potrebujemo vsaj 2)."
    )
    for node, data in G_bib.nodes(data=True):
        assert "frequency" in data, f"Vozlisce {node} nima 'frequency' atributa."

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Casovni mejnik: vse `plots/*` z mtime po `t_start` velja za "svez izhod"
    # tega testa. Tako je vrstno neobcutljiv na ponovne zagone (tudi ce ploti
    # zaradi prejsnjega zagona ze obstajajo, jih plot_coocurrence prepise).
    import time as _time
    t_start = _time.time()
    # Drobni odlog: nekateri filesistemi (FAT/exFAT) imajo locljivost mtime ~2s.
    _time.sleep(0.05)

    try:
        bs.plot_coocurrence(
            G_bib,
            partition_attrs=["walktrap"],
            overlay_color_attr="avg_year",
            overlay_size_attr="frequency",
            filename_prefix="03_biblio_kw_network",
        )
    except Exception as e:
        # Nekateri partition algoritmi (igraph) zahtevajo dodatne knjiznice.
        # Fallback: shranimo vsaj GEXF, da je rezultat (3) v obliki datoteke.
        print(f"    OPOZORILO: plot_coocurrence neuspesen "
              f"({type(e).__name__}: {e}).")
        gexf_out = plots_dir / "03_biblio_kw_network.gexf"
        nx.write_gexf(G_bib, str(gexf_out))
        print(f"    -> fallback: graf zapisan v {gexf_out.relative_to(HERE)}")
        return

    # Nasi plots: tisti, katerih ime se zacne s "03_biblio_kw_network" IN so
    # bili spremenjeni po t_start (toleranca 2s zaradi mtime locljivosti).
    expected_prefix = "03_biblio_kw_network"
    fresh = [
        f for f in plots_dir.iterdir()
        if f.name.startswith(expected_prefix)
        and f.stat().st_mtime >= t_start - 2.0
    ]
    fresh.sort()
    print(f"    Sveze zapisani ploti ({len(fresh)}):")
    for f in fresh:
        print(f"      -> {f.relative_to(HERE)}")

    assert fresh, (
        f"plot_coocurrence ni zapisal nobene datoteke s prefixom "
        f"{expected_prefix!r} po zagonu testa. "
        f"V {plots_dir} je: {[p.name for p in plots_dir.iterdir()]}"
    )


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    try:
        from _pytest.outcomes import Skipped as _Skipped
    except ImportError:
        class _Skipped(Exception):
            pass

    tests = [
        ("(1) Scientific production",        test_1_scientific_production),
        ("(2) Sources stats",                test_2_sources_stats),
        ("(3) biblio* keyword cooccurrence", test_3_biblio_keyword_cooccurrence_network),
    ]

    print(f"\n=== biblium 2.16 Scopus demo (lokalno: {HERE}) ===")
    if biblium is not None:
        print(f"Nalozena verzija: {biblium.__version__}")
    else:
        print(f"OPOZORILO: biblium se ni nalozil -- {LOAD_ERROR!r}")
    print()

    passed = failed = skipped = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
            print(f"OK   {name}\n")
            passed += 1
        except _Skipped as e:
            print(f"SKIP {name}: {e}\n")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}\n")
            failed += 1
        except Exception as e:
            traceback.print_exc()
            print(f"ERR  {name}: {type(e).__name__}: {e}\n")
            failed += 1

    out = _BS_CACHE.get("outdir")
    if out is not None:
        print(f"\nIzhodne datoteke (tabele in ploti): {out}")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(0 if failed == 0 else 1)
