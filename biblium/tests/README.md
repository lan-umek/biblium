# Biblium test suite

This directory holds the regression tests for biblium. The suite is
designed to be:

- **fast** (the full run takes well under 10 seconds),
- **deterministic** (every RNG state is seeded explicitly),
- **hermetic** (no test makes a live HTTP request).

## Running the tests

From the biblium root directory:

```bash
# Full suite
pytest

# Single module
pytest biblium/tests/test_cobiss_parser.py

# Single test class or method
pytest biblium/tests/test_firth.py::TestMLELogit
pytest biblium/tests/test_firth.py::TestMLELogit::test_coefficients_match_statsmodels

# Verbose output with one line per test
pytest -v

# Show local variables on failure
pytest -l
```

Markers are declared in `pyproject.toml`. To skip slow tests (the suite
contains none today, but the marker is there for future use):

```bash
pytest -m "not slow"
```

## File layout

```
tests/
├── conftest.py                   # shared fixtures (RNG, paths, synthetic data)
├── fixtures/
│   ├── cobiss_sample.md          # 7-record COBISS+ sample (markdown form)
│   └── cobiss_full_sample.md     # 14-record sample with edge cases
├── test_cobiss_typology.py       # typology table + label/code lookups
├── test_cobiss_parser.py         # parser on real-shape COBISS samples
├── test_cobiss_api.py            # HTTP client without live network
├── test_readbib_cobiss.py        # read_bibfile(db="cobiss") dispatcher
├── test_permutation.py           # permutation_test, adjust_p_values, ...
├── test_firth.py                 # mle_logit, firth_logit, fit_logit
└── test_associations_perm.py     # integration: chi2 + Firth on doc-term
```

## Adding a new test

The standard recipe is one small file per module being tested, with
classes that group related assertions. For example, `TestAuthorParsing`
inside `test_cobiss_parser.py` collects every test about author
extraction.

When adding a fixture, prefer to put it in `conftest.py` if more than
one test module uses it; keep it inside the test file otherwise.

## Hermetic-ness

`test_cobiss_api.py` patches `CobissClient.fetch` to return canned
fixtures rather than making any HTTP call. This is intentional: the
COBISS+ system is a small public service run by IZUM, and we should
not exercise it from CI. If you need to validate a parser change
against a real personal bibliography, run the manual smoke script:

```python
from biblium.cobiss_api import fetch_personal_bibliography_to_csv
result = fetch_personal_bibliography_to_csv(
    "https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_<...>.html",
    "out.csv",
)
print(f"{result.n_records} records for {result.researcher_name}")
```
