# -*- coding: utf-8 -*-
"""
Tests for ``biblium.term_extraction`` (new in 2.19).

Tiny synthetic texts with hand-computed expectations; deterministic and
hermetic.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from biblium import term_extraction as T


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------

class TestPhrases:
    SENT = "Bibliometric analysis of public management reform using the Scopus database"

    def test_chunks(self):
        chunks = T._chunks(self.SENT, T.STOPWORDS)
        assert chunks == [["bibliometric", "analysis"], ["public", "management", "reform"],
                          ["scopus", "database"]]

    def test_ngrams_max3(self):
        got = T.extract_candidate_phrases(self.SENT, max_len=3)
        assert got == [
            "bibliometric", "bibliometric analysis", "analysis",
            "public", "public management", "public management reform",
            "management", "management reform", "reform",
            "scopus", "scopus database", "database",
        ]

    def test_min_len(self):
        got = T.extract_candidate_phrases(self.SENT, max_len=3, min_len=2)
        assert got == ["bibliometric analysis", "public management", "public management reform",
                       "management reform", "scopus database"]

    def test_sentence_and_digit_splits(self):
        got = T.extract_candidate_phrases("Citation networks. Social networks in 2020 studies", max_len=3)
        # "." splits the sentence, "2020" splits the chunk; last words singularised
        assert "citation network" in got and "social network" in got
        assert "network social" not in got and "networks social" not in got
        assert "study" in got and "network study" not in got

    def test_hyphens_and_case(self):
        got = T.extract_candidate_phrases("Co-citation Analysis", max_len=2)
        assert got == ["co-citation", "co-citation analysis", "analysis"]

    def test_custom_stopwords(self):
        got = T.extract_candidate_phrases("alpha beta gamma", max_len=3, stopwords=["beta"])
        assert got == ["alpha", "gamma"]

    def test_empty_and_nan(self):
        assert T.extract_candidate_phrases("") == []
        assert T.extract_candidate_phrases(np.nan) == []
        assert T.extract_candidate_phrases(None) == []

    def test_bad_lengths(self):
        with pytest.raises(ValueError):
            T.extract_candidate_phrases("x", max_len=1, min_len=2)


class TestSingularize:
    @pytest.mark.parametrize("plural, singular", [
        ("studies", "study"), ("countries", "country"),
        ("processes", "process"), ("boxes", "box"), ("approaches", "approach"), ("wishes", "wish"),
        ("cases", "case"), ("databases", "database"), ("networks", "network"),
        ("analyses", "analysis"), ("indices", "index"), ("data", "data"), ("series", "series"),
        ("bibliometrics", "bibliometrics"), ("class", "class"), ("bus", "bus"), ("thesis", "thesis"),
        ("gas", "gas"),  # length <= 3: unchanged
        ("co-authorships", "co-authorship"),
        ("", ""),
    ])
    def test_rules(self, plural, singular):
        assert T.singularize(plural) == singular


# ---------------------------------------------------------------------------
# Term-document matrix
# ---------------------------------------------------------------------------

@pytest.fixture
def docs():
    return pd.DataFrame({
        "Title": [
            "Citation network analysis of citation networks",
            "Citation network of journals",
            "Journals and networks",
            "Keyword co-occurrence",
        ],
        "Abstract": [np.nan, np.nan, np.nan, np.nan],
    })


class TestTDM:
    def test_min_occurrences_no_merge(self, docs):
        X, terms = T.build_term_document_matrix(docs, min_occurrences=2, merge_nested=False,
                                                drop_generic=False)
        assert terms == ["citation", "citation network", "journal", "network"]
        assert X.shape == (4, 4)
        M = X.toarray()
        assert M[:, terms.index("citation")].tolist() == [1, 1, 0, 0]
        assert M[:, terms.index("network")].tolist() == [1, 1, 1, 0]
        assert M[:, terms.index("journal")].tolist() == [0, 1, 1, 0]
        assert M.max() == 1  # binary even though doc 0 contains "citation" twice

    def test_counts_when_not_binary(self, docs):
        X, terms = T.build_term_document_matrix(docs, min_occurrences=2, merge_nested=False,
                                                drop_generic=False, binary=False)
        M = X.toarray()
        assert M[0, terms.index("citation")] == 2
        assert M[0, terms.index("network")] == 2
        assert "citation network" in terms  # 3 occurrences in total

    def test_merge_nested(self, docs):
        X, terms = T.build_term_document_matrix(docs, min_occurrences=2, merge_nested=True,
                                                drop_generic=False)
        # "citation" always occurs inside "citation network" -> dropped;
        # "network" stands alone in doc 2 (3 of 4 occurrences nested) -> kept
        assert terms == ["citation network", "journal", "network"]

    def test_min_docs_and_drop_generic(self):
        d = pd.DataFrame({"Title": ["Panel data study", "Panel data study results", "Study results"]})
        X, terms = T.build_term_document_matrix(d, min_occurrences=2, merge_nested=False)
        assert "study" not in terms and "result" not in terms and "study result" not in terms
        assert "panel data" in terms and "data" not in terms  # "data" alone is generic
        X2, terms2 = T.build_term_document_matrix(d, min_occurrences=2, min_docs=3, merge_nested=False)
        assert terms2 == []
        assert X2.shape == (3, 0)

    def test_copyright_and_no_abstract_stripped(self):
        d = pd.DataFrame({"Title": ["Citation network", "Citation network"],
                          "Abstract": ["[No abstract available]", "Great paper. © 2020 Elsevier B.V. All rights reserved."]})
        X, terms = T.build_term_document_matrix(d, min_occurrences=1, drop_generic=False, merge_nested=False)
        assert "elsevier" not in terms and "abstract available" not in terms
        X2, terms2 = T.build_term_document_matrix(d, min_occurrences=1, drop_generic=False,
                                                  merge_nested=False, strip_copyright=False)
        assert "elsevier" in terms2

    def test_missing_columns(self, docs):
        with pytest.raises(ValueError):
            T.build_term_document_matrix(docs, columns=("Nope",))


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

class TestRelevance:
    @pytest.fixture
    def X(self):
        # docs: {g,a}, {g,b}, {g,c}, {a,c}, {a,c}; terms g (generic), a, b, c
        rows = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], float)
        return sp.csr_matrix(rows), ["g", "a", "b", "c"]

    def test_values_and_ordering(self, X):
        X, terms = X
        s = T.relevance_scores(X, terms).set_index("Term")
        # co-occurrence: g-a 1, g-b 1, g-c 1, a-c 2; q = (g .3, a .3, b .1, c .3)
        rel_g = (1 / 3) * math.log((1 / 3) / 0.3) * 2 + (1 / 3) * math.log((1 / 3) / 0.1)
        rel_b = math.log(1 / 0.3)
        rel_a = (1 / 3) * math.log((1 / 3) / 0.3) + (2 / 3) * math.log((2 / 3) / 0.3)
        assert s.loc["g", "Relevance"] == pytest.approx(rel_g)
        assert s.loc["b", "Relevance"] == pytest.approx(rel_b)
        assert s.loc["a", "Relevance"] == pytest.approx(rel_a)
        assert s.loc["c", "Relevance"] == pytest.approx(rel_a)
        # the uniformly co-occurring term is the least relevant
        assert s.loc["g", "Relevance"] < s.loc["a", "Relevance"] < s.loc["b", "Relevance"]
        assert s.loc["g", "Relevance rank"] == 4 and s.loc["b", "Relevance rank"] == 1
        assert s.loc["g", "Occurrences"] == 3 and s.loc["a", "Occurrences"] == 3

    def test_isolated_term_zero(self):
        X = sp.csr_matrix(np.array([[1, 0], [0, 1]], float))
        s = T.relevance_scores(X, ["x", "y"])
        assert (s["Relevance"] == 0).all()

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            T.relevance_scores(sp.csr_matrix((2, 2)), ["x"])

    def test_select_terms_share(self):
        scores = pd.DataFrame({"Term": list("abcdefghij"), "Occurrences": 5,
                               "Relevance": np.arange(10, 0, -1, dtype=float),
                               "Relevance rank": np.arange(1, 11)})
        sel = T.select_terms(scores, share=0.6)
        assert sel["Term"].tolist() == list("abcdef")
        sel2 = T.select_terms(scores, share=0.6, min_relevance=8)
        assert sel2["Term"].tolist() == list("abc")
        assert len(T.select_terms(scores, share=0.25)) == 3  # ceil(2.5)
        with pytest.raises(ValueError):
            T.select_terms(scores, share=0)


# ---------------------------------------------------------------------------
# Pipeline, co-occurrence, column, plot
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.fixture
    def corpus(self):
        return pd.DataFrame({
            "Title": ["Citation network of journals", "Citation network of authors",
                      "Keyword co-occurrence of authors", "Keyword co-occurrence network",
                      "Journals and authors"],
            "Abstract": ["Citation network methods.", "Authors cite journals.", np.nan,
                         "Keyword maps.", "Authors publish in journals."],
        })

    def test_extract_terms(self, corpus):
        sel, X, terms = T.extract_terms(corpus, min_occurrences=2, share=1.0, drop_generic=False)
        assert sorted(sel["Term"]) == terms
        assert X.shape == (5, len(terms))
        assert {"citation network", "author", "journal", "keyword co-occurrence"} <= set(terms)
        sel60, X60, terms60 = T.extract_terms(corpus, min_occurrences=2, share=0.6, drop_generic=False)
        assert len(terms60) == math.ceil(0.6 * len(terms))
        assert X60.shape[1] == len(terms60)
        # columns of X_sel are the selected columns of the full matrix
        full_X, full_terms = T.build_term_document_matrix(corpus, min_occurrences=2, drop_generic=False)
        for j, t in enumerate(terms60):
            assert np.array_equal(X60[:, j].toarray(), full_X[:, full_terms.index(t)].toarray())

    def test_cooccurrence_symmetric_with_diagonal(self, corpus):
        X, terms = T.build_term_document_matrix(corpus, min_occurrences=2, drop_generic=False)
        co = T.term_cooccurrence(X, terms)
        assert list(co.index) == terms and list(co.columns) == terms
        assert (co.values == co.values.T).all()
        occ = np.asarray(X.sum(axis=0)).ravel()
        assert np.array_equal(np.diag(co.values), occ)
        assert co.loc["author", "journal"] == 2  # docs 1 and 4
        assert co.loc["citation network", "keyword co-occurrence"] == 0

    def test_add_terms_column(self, corpus):
        X, terms = T.build_term_document_matrix(corpus, min_occurrences=2, drop_generic=False)
        out = T.add_terms_column(corpus, X, terms)
        assert "Extracted Terms" in out.columns and len(out) == len(corpus)
        assert "citation network" in out.loc[0, "Extracted Terms"].split("; ")
        assert out.loc[4, "Extracted Terms"] == "author; journal"
        assert "Extracted Terms" not in corpus.columns  # original untouched
        with pytest.raises(ValueError):
            T.add_terms_column(corpus, X, terms[:-1])

    def test_plot(self, corpus, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel, X, terms = T.extract_terms(corpus, min_occurrences=2, share=1.0, drop_generic=False)
        out = tmp_path / "rel.png"
        fig, ax = T.plot_term_relevance(sel, top_n=5, f_name=str(out), dpi=50)
        assert out.exists() and len(ax.patches) == min(5, len(sel))
        plt.close(fig)
