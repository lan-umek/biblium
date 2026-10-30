# Changelog

## 2.19.1 (2026-10-30)

### New
- `biblium.merge`: combine exports from several databases (Scopus, WoS, OpenAlex, PubMed, Dimensions, Lens, …) into one de-duplicated dataset. Matching by DOI (default), by normalised title, or DOI with title fallback; column policy `union` (all fields) or `intersection` (fields common to all sources); per-source citation counts, union of keywords/references, longest abstract, source priority; provenance columns `Source DB`, `N sources`, `Merge key`, `Match type`; overlap and column-coverage reports.
- `BiblioStats.from_databases(sources, key=..., columns=...)` and `BiblioStats.merge_with(sources, ...)`; `methods_text()` describes the merge.
- GUI: *Data → Merge Databases* panel (file pickers per source, matching options, overlap tables/plot, loads the merged dataset).
- `biblium.datasets`: catalogue of 47 OpenAlex demo datasets (~200 works each) covering public administration (8 topics incl. a Slovenian sample), medicine, sociology, political science, economics, business, education, psychology, law, computer science, mathematics, environmental and life sciences, information science, plus 12 journal-specific samples (PAR, JPART, PMR, CEPAR, Scientometrics, JOI, ASR, AER, The Lancet, Harvard Law Review, CACM, Annals of Mathematics). `fetch_dataset`/`fetch_all` download through the OpenAlex API and write website-export-compatible `.csv.gz` files; `load_dataset(name)` returns a `BiblioStats`; CLI `python -m biblium.datasets --fetch-all --email …`; `methods_text()` describes the dataset. GUI: *Load Data → Domain Datasets* card (domain/dataset pickers, Use dataset, Download missing…).

### Fixed
- `beautifulsoup4` and `lxml` added to core dependencies (COBISS HTML parsing); SPDX license metadata.
- GUI imports without a TkAgg backend (lazy annotations in `plots.py`); `BasePanel._show_no_data_message` restored (seven panels crashed when opened without data).
- Firth test fixture was collinear (profile-CI test depended on the scipy version).
- `readbib`: numeric coercion in the OpenAlex/Dimensions/Lens readers did not run under pandas ≥ 3 (string dtype is `str`, not `object`), so `Cited by`/`Year` stayed text.

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
