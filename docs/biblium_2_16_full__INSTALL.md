# Biblium 2.16 — Installation and Quick Start

This archive contains the full source tree for **biblium 2.16**.

## What's in 2.16 (vs 2.13)

Three major feature additions plus a comprehensive test suite:

### 1. Permutation inference (`biblium.utilsbib_modules.permutation`)

- `permutation_test()` — core engine, row-permutes group matrix while
  preserving marginals and overlaps
- `assoc_permutation_test()` — DataFrame-friendly wrapper supporting
  `test=("chi2"|"cramers_v"|"total_inertia"|"dimension_inertias"|"residuals")`
- `adjust_p_values()` — BH / Holm / Bonferroni correction, matches
  `scipy.stats.false_discovery_control` to 1e-15
- Phipson-Smyth formula `p = (1 + #{T_b ≥ T_0}) / (1 + B)` (avoids spurious
  `p = 0`)
- Adaptive permutation budget via micro-benchmark
- Disjoint-groups detection with `UserWarning`

### 2. Firth penalised logistic regression (`biblium.utilsbib_modules.firth`)

- `firth_logit()` — Jeffreys-prior penalisation, profile-likelihood CIs
- `mle_logit()` — matches `statsmodels.Logit` to 5e-17
- `fit_logit(method="auto"|"firth"|"mle")` — automatic fallback to Firth
  on ill-conditioned designs or post-MLE separation detection
- LRT p-values via `compute_lrt=True`
- `BiblioGroupClassifier.logistic_regression_analysis` is a drop-in
  replacement for the old statsmodels-based version

### 3. COBISS+ support (`biblium.cobiss_api`, `biblium.utilsbib_modules.cobiss_*`)

- HTTP client with rate limit, retry/backoff, on-disk cache
- Three URL families:
    - **Pre-rendered HTML** (`/bibliographies/<country>/webBiblio/bib201_*.html`)
    - **CRIS direct link** (`/biblioweb/direct/<country>/<lang>/cris/<code>`)
    - **CONOR direct link** (`/biblioweb/direct/<country>/<lang>/conor/<id>`)
- Direct links automatically request `?format=X&citation=true` (XML, with
  WoS/Scopus citation counts) when run from a `.si` IP
- `parse_cobiss_html()` — robust regex/BeautifulSoup parser tested on
  edge cases: `et al.`, multi-word surnames, apostrophes, role tags
- `parse_cobiss_xml()` — skeleton parser awaiting a real XML sample
  for the per-record element names
- `read_bibfile(f, db="cobiss")` integration; `read_cobiss_html(f)` shortcut
- COBISS typology mapping (50 codes from the official COBISS revision
  29 September 2025)

### 4. Test suite (`biblium/tests/`)

213 tests, all passing in ~10 seconds:

```
test_associations_perm.py        5 tests
test_cobiss_api.py              33 tests
test_cobiss_parser.py           44 tests
test_cobiss_routing.py          11 tests
test_cobiss_typology.py         33 tests
test_cobiss_url.py              24 tests
test_cobiss_xml_parser.py       10 tests
test_firth.py                   21 tests
test_permutation.py             26 tests
test_readbib_cobiss.py           6 tests
```

## Installation

### Option A — Drop-in replacement of an existing biblium 2.13

1. Locate your existing biblium installation, e.g.
   `C:\Users\Lan\OneDrive - Univerza v Ljubljani\aplikacije\biblium\`.
2. Back up the existing folder (rename to `biblium_2_13_backup`).
3. Extract this archive so the `biblium/` folder lands at the same path.

### Option B — Editable install in a fresh project

```bash
unzip biblium_2_16_full.zip -d C:\projects\biblium_2_16
cd C:\projects\biblium_2_16
pip install -e .
```

## Dependencies

Required:

```
pandas, numpy, scipy, scikit-learn, matplotlib,
networkx, requests, beautifulsoup4, lxml, NLTK
```

Optional (degraded functionality if missing):

```
statsmodels      # for the MLE comparison; firth_logit works without it
PySide6          # GUI (BibScientia)
sentence_transformers, torch    # for embeddings-based features
```

## Running the tests

```bash
cd biblium_2_16_root
pytest biblium/tests/                      # all tests
pytest biblium/tests/test_cobiss_parser.py # one module
pytest -v                                  # verbose
```

Expected: **213 passed in ~10s**.

## Quick start — COBISS

### Pre-rendered HTML URL (the typical case)

You generate this URL by filling in the personal-bibliography form on
COBISS+ and copying the resulting URL.

```python
from biblium.cobiss_api import fetch_personal_bibliography_to_csv

result = fetch_personal_bibliography_to_csv(
    "https://bib.cobiss.net/bibliographies/si/webBiblio/"
    "bib201_20260430_111142_28519.html",
    "umek_bibliography.csv",
)
print(f"{result.n_records} records for {result.researcher_name} ({result.period})")
print(f"Format used: {result.format_used}")  # "html" for pre-rendered URLs
```

### Direct link by ARIS researcher's code

```python
result = fetch_personal_bibliography_to_csv(
    "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
    "umek_2024.csv",
    # prefer_format="auto" by default -- requests XML, falls back to HTML
)
```

The client automatically rewrites the URL to
`?format=X&citation=true` to request the structured XML form (which is
much more robust against UI changes than HTML scraping). Until the
XML parser's record-level extraction is wired up against a real COBISS
XML sample, the client silently falls back to HTML parsing on the
same response.

### Loading the CSV back into biblium

```python
from biblium.bibstats import BiblioStats

bs = BiblioStats(f_name="umek_bibliography.csv", db="cobiss")
print(bs.df["Document Type"].value_counts())
```

## Quick start — Permutation tests in `associate_items`

```python
from biblium.bibstats import BiblioStats
from biblium.bibclass import BiblioGroupClassifier

bs = BiblioStats(f_name="my_scopus.csv", db="scopus")
gc = BiblioGroupClassifier(bs, group_membership="Year")  # 3 disjoint year buckets

# Old-style chi-square (asymptotic) is still the default.
# Permutation chi2 + BH correction over standardised residuals:
result = gc.associate_items(
    text_column="Index Keywords",
    inference="permutation",
    permutation_tests=("chi2", "residuals"),
    n_permutations=4999,
    multiple_testing="bh",
    random_state=2026,
)
```

## Quick start — Firth logit on (near-)separated data

```python
from biblium.utilsbib_modules.firth import fit_logit

res = fit_logit(X, y, method="auto")   # MLE if well-conditioned, Firth otherwise
print(res.method_used)                 # "mle" | "firth" | "firth_after_mle_failure"
print(res.coef, res.se, res.p_values_wald)
print(res.to_statsmodels_compat_summary())   # statsmodels-style table
```

## What's NOT yet implemented in 2.16

- **CRIS XML record-level parsing.** The XML parser skeleton is in place
  and tested for document metadata + dispatch, but per-record element
  extraction (`_parse_unit()`) raises `NotImplementedError` until
  IZUM's actual response shape is verified against a real `format=X`
  sample. Until then, the client silently falls back to HTML parsing
  for direct-link URLs as well.

- **COBISS+ live HTTP from this Claude session.** Anthropic's proxy
  blocks `bib.cobiss.net` at the network layer (returns
  `"Host not in allowlist"`). All client-level logic was tested
  against monkey-patched fetch responses; running against the live
  site happens on your machine.

- **Slovenian-language UI option for typology labels.** The mapping
  table contains both Slovenian and English labels for all 50 codes;
  a `lang="sl"` parameter through to `read_cobiss_html` is a one-liner
  to add when needed.

## Author / contact

Lan Umek — Faculty of Public Administration, University of Ljubljana
