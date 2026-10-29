# biblium

**biblium** is a Python library (and a desktop GUI) for bibliometric and scientometric analysis of
Scopus, Web of Science, OpenAlex, PubMed, Dimensions, Lens.org and COBISS exports.

[![PyPI](https://img.shields.io/pypi/v/biblium.svg)](https://pypi.org/project/biblium/)
[![Tests](https://github.com/lan-umek/biblium/actions/workflows/tests.yml/badge.svg)](https://github.com/lan-umek/biblium/actions/workflows/tests.yml)

## Installation

```bash
pip install biblium                 # core
pip install "biblium[network]"      # + igraph / louvain partitions
pip install "biblium[full]"         # everything (LLM, NLP, geo, interactive plots, ML)
```

Python 3.10 or newer is required.

## Quick start

```python
from biblium import BiblioStats

bib = BiblioStats(f_name="scopus_export.csv", db="scopus")      # res_folder=None: nothing is written to disk
bib.get_main_info()
bib.count_sources(); bib.get_sources_stats()

# 2.19 indicators
bib.citation_css(by_year=True)                 # Characteristic Scores and Scales
bib.percentile_indicators("Source title")      # PP(top 10 %), PP(top 1 %), MNCS
bib.reference_age()                            # Price index, citing half-life
bib.concentration("Authors", plot=True)        # Lorenz curve, Gini, Theil
bib.rpys_patterns()                            # RPYS + hot papers / sleeping beauties / life cycles
bib.dual_map(plot=True)                        # citing vs cited disciplines (ASJC)
bib.extract_terms()                            # VOSviewer-style terms with relevance scores
print(bib.methods_text("en"))                  # reproducible "Methods" paragraph with references
```

Pass `res_folder="results"` to save every table (xlsx) and figure (png/svg/pdf) automatically.

## Group analysis

```python
from biblium import BiblioGroupAnalysis
grp = BiblioGroupAnalysis(f_name="scopus_export.csv", db="scopus", group_matrix=groups)
grp.association_tests(permutations=999)
```

## Desktop application

```bash
biblium-gui
```

The Tkinter application exposes the whole library in ~90 panels (data loading and API harvesting,
counts and statistics, bibliometric laws, networks, thematic maps and evolution, topic models,
group analysis, PRISMA, metadata completion, journal rankings, funding, RPYS, dual-map overlay,
snowballing, term extraction, reports).

## Citing

Umek, L. (2026). Biblium: a Python library for comparative bibliometric analysis. *Scientometrics*.
https://doi.org/10.1007/s11192-026-05636-8

## License

MIT
