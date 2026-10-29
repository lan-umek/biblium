# Changelog

## 2.19.0 (2026-10-29)

### New analyses (`biblium.indicators`, each with a `BiblioStats` method and a GUI panel)
- Characteristic Scores and Scales (Glänzel & Schubert 1988): `citation_css()`
- Leiden-style percentile indicators per entity (PP top 10 %, PP top 1 %, MNCS): `percentile_indicators()`
- Reference age, Price index, citing half-life: `reference_age()`
- Cited sources ("knowledge base") with overlap statistics: `cited_sources()`
- Concentration (Lorenz curve, Gini, Theil, HHI): `concentration()`
- Author careers (academic age, m-quotient, local h): `author_careers()`
- Uncitedness by year/group: `uncitedness()`
- Heaps' law of vocabulary growth: `fit_heaps_law()`
- Bootstrap confidence intervals for indicators: `indicator_ci()`

### Gap-closing features vs. bibliometrix / CiteSpace / VOSviewer
- PRISMA 2020 flow diagram (`prisma.py`)
- Missing-data audit and metadata completion by DOI from OpenAlex/Crossref with provenance and undo (`metadata_completion.py`)
- Journal ranking lists (SCImago/JCR/ABS/ABDC) matched by ISSN or title, quartile summaries and filtering (`journal_rankings.py`)
- Funding analysis: funders, trend, funded vs. unfunded impact (`funding.py`)
- RPYS with centered/backward median and cited-reference pattern classification (`reference_patterns.py`)
- Diachronic co-word networks with animated GIF, fuzzy document-cluster membership, CiteSpace-style cluster timeline, structural variation analysis (`network_dynamics.py`)
- Dual-map overlay of citing vs. cited disciplines via Scopus ASJC (`dual_map.py`)
- Citation snowballing / corpus expansion through OpenAlex (`snowball.py`)
- VOSviewer-style term extraction with relevance scores (`term_extraction.py`)
- `methods_text()` covers all new analyses (English/Slovene, with references)

### GUI
- The `biblium.gui` package is importable again (missing `gui.panels.reports` and `TopicsPanel` restored); `biblium-gui` console script works
- `SimpleAnalysisPanel` base and 22 new panels (sections Indicators, References, Dynamics & Text, Review & Quality)
- Results folders are opt-in: `res_folder=None` by default, nothing is written to disk unless requested

### Fixes and merges
- Merged post-2.16 changes (optional statsmodels in `bibclass`, `edition_fingerprints`, `utilsbib_modules.editions`, richer `dashboard_builder`) with the 2.18 consolidation
- 2.18 fixes carried over: tapered h-index, π-index (Vinkler), RPYS, local citations, thematic evolution, main path layout, bursts, igraph partitions, etc. (see POROCILO files)
- plotly and huggingface_hub remain optional (import guards)
- Test suite: 497 tests

### Requirements
- Python >= 3.10 (was 3.9)
