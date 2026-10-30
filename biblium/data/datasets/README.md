# biblium demo datasets (OpenAlex)

Each `<name>.csv.gz` in this folder is a sample of ~200 works downloaded from the
OpenAlex API (CC0 metadata) with `python -m biblium.datasets`. Files use the
OpenAlex website-export column layout (dot-notation columns, `|`-joined lists),
so they load with `db="openalex"` like any OpenAlex export:

```python
from biblium import datasets
print(datasets.list_datasets())              # catalogue
bib = datasets.load_dataset("pa_egovernment")  # BiblioStats
```

Download / refresh (needs internet; use your e-mail for the polite pool):

```
python -m biblium.datasets --list
python -m biblium.datasets --fetch-all --email you@example.org
python -m biblium.datasets --fetch pa_egovernment j_jpart --email you@example.org
```

The catalogue (query specs) lives in `biblium/datasets.py`.
