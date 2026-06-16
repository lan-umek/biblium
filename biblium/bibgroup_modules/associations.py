# -*- coding: utf-8 -*-
"""
Group association methods for BiblioGroup.

This module provides mixin methods for associating groups with various entities.

New in 2.16
-----------
``associate_items`` (and the ``associate_*`` wrappers it delegates to) now
accept three additional parameters:

- ``inference`` -- ``"asymptotic"`` (default), ``"permutation"`` or ``"both"``.
  When permutation inference is requested, a row-wise permutation test is
  run on the contingency table after the standard analysis. Permutation
  results are attached to the resulting ``Relation`` object as
  ``permutation_*`` attributes (see Notes).
- ``n_permutations`` -- number of permutations (None triggers an adaptive
  choice from a target time budget).
- ``multiple_testing`` -- correction for cell-level residual p-values
  (``"bh"``, ``"holm"``, ``"bonferroni"``, ``"none"``).

A separate parameter:

- ``logit`` -- ``"none"`` (default), ``"firth"``, ``"auto"``, or ``"mle"``.
  When set, a per-group binary logit (group membership ~ entity
  indicators) is fitted with the chosen method. Results are stored on
  the ``Relation`` object as ``R.logit_results`` (a dict keyed by group).

Notes
-----
Permutation results live on the returned ``Relation`` object as:

- ``R.permutation_chi2_p`` -- global chi-squared p-value
- ``R.permutation_chi2_p_ci`` -- 95% Clopper-Pearson CI for that p-value
- ``R.permutation_residuals_p`` -- per-cell two-sided p-values, same
  shape as ``R.chi2_residuals_df``
- ``R.permutation_residuals_p_adj`` -- multiple-testing-adjusted p-values
- ``R.permutation_n`` -- number of permutations actually performed
- ``R.permutation_disjoint`` -- whether ``group_matrix`` is a disjoint
  partition (in which case the asymptotic test is unbiased and the
  permutation test is informational)
"""

from __future__ import annotations

import os
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from biblium import utilsbib

if TYPE_CHECKING:
    from biblium.bibgroup import BiblioGroup


# ---------------------------------------------------------------------------
# Lazy imports for the new inference machinery
# ---------------------------------------------------------------------------

def _get_permutation_module():
    """Lazy import of the permutation submodule."""
    from biblium.utilsbib_modules import permutation as _perm
    return _perm


def _get_firth_module():
    """Lazy import of the firth (logistic regression) submodule."""
    from biblium.utilsbib_modules import firth as _firth
    return _firth


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class GroupAssociationsMixin:
    """Mixin providing associate_* methods for BiblioGroup."""

    # Type hints for attributes from BiblioGroup
    df: pd.DataFrame
    group_matrix: pd.DataFrame
    default_separator: str
    res_folder: Optional[str]

    def associate_items(
        self: BiblioGroup,
        *,
        domain_key: str,
        item_col: str,
        count_type: str = "single",
        value_type: str = "string",
        item_column_name: str = "Item",
        translated_column_name: Optional[str] = None,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        # --- new in 2.16: permutation inference --------------------------
        inference: Literal["asymptotic", "permutation", "both"] = "asymptotic",
        n_permutations: Optional[int] = None,
        target_seconds: float = 5.0,
        multiple_testing: Literal["bh", "holm", "bonferroni", "none"] = "bh",
        permutation_tests: Tuple[str, ...] = ("chi2", "residuals"),
        random_state: Optional[int] = None,
        # --- new in 2.16: Firth-penalised logistic regression ------------
        logit: Literal["none", "firth", "auto", "mle"] = "none",
        logit_ci: Optional[Literal["wald", "profile"]] = None,
        logit_compute_lrt: bool = False,
        # -----------------------------------------------------------------
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Build item indicators and relate them to groups.

        Workflow:
        1. Count items via utilsbib.count_occurrences.
        2. Select items by items_included/top_n/min_freq and apply items_excluded.
        3. Build indicators via utilsbib.match_items_and_compute_binary_indicators.
        4. Relate groups to items using self.group_matrix.
        5. (NEW) If ``inference != "asymptotic"`` run permutation tests
           and attach results to the ``Relation`` object.
        6. (NEW) If ``logit != "none"`` fit a per-group Firth/MLE logit
           and attach the results dict to the ``Relation`` object.
        7. If ``filename`` is provided, save associations to Excel.

        Parameters
        ----------
        domain_key : str
            Key to identify this association domain.
        item_col : str
            Column name containing the items.
        count_type : str, default "single"
        value_type : str, default "string"
        item_column_name : str, default "Item"
        translated_column_name : str, optional
        include_stats : tuple, default ("diversity", "correspondence", "chi2", "svd", "log-ratio")
            Asymptotic / descriptive statistics to compute. Permutation
            p-values for these statistics are added separately when
            ``inference != "asymptotic"``.
        clean_zeros : bool, default True
        to_self : bool, default False
        top_n : int, default 0
        min_freq : int, default 5
        items_included, items_excluded : list or str, optional

        inference : {"asymptotic", "permutation", "both"}, default "asymptotic"
            - "asymptotic": classical chi-squared and CA. Unbiased only
              when groups are disjoint AND entities are single-valued.
            - "permutation": replace asymptotic p-values with
              permutation p-values, valid even when groups overlap or
              entities are multi-valued.
            - "both": keep asymptotic statistics AND add permutation
              p-values for direct comparison (recommended for
              methodological reporting).
        n_permutations : int, optional
            Number of permutations. ``None`` -> adaptive choice based on
            ``target_seconds``.
        target_seconds : float, default 5.0
            Time budget for adaptive ``n_permutations``.
        multiple_testing : {"bh", "holm", "bonferroni", "none"}, default "bh"
            Correction applied to cell-level residual p-values.
        permutation_tests : tuple of str, default ("chi2", "residuals")
            Which permutation tests to run. Subset of:
            ``"chi2"``, ``"cramers_v"``, ``"total_inertia"``,
            ``"dimension_inertias"``, ``"residuals"``.
        random_state : int, optional
            Seed for reproducible permutation results.

        logit : {"none", "firth", "auto", "mle"}, default "none"
            - "none" : do not fit a logistic regression.
            - "firth": always fit Firth-penalised logit.
            - "auto" : detect (near-)singular design or post-hoc
              separation; fall back to Firth otherwise use MLE.
            - "mle"  : standard MLE (no separation handling).
        logit_ci : {"wald", "profile"}, optional
            Defaults to "profile" for Firth, "wald" for MLE.
        logit_compute_lrt : bool, default False
            Compute likelihood-ratio p-values (one constrained refit per
            coefficient -- expensive).

        filename : str, optional
        **kwargs :
            Additional options forwarded to ``count_occurrences`` (e.g. ``sep``).

        Returns
        -------
        BiblioGroup
            Self for method chaining. The relation is stored in
            ``self.<domain_key>_associations`` (a ``Relation`` object,
            possibly with ``permutation_*`` and ``logit_*`` attributes)
            and the contingency table in ``self.<domain_key>_contingency``.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix is missing. Build groups first.")

        # Extract sep for count_occurrences
        sep_for_count = kwargs.pop("sep", None)
        if sep_for_count is None:
            sep_for_count = getattr(self, "default_separator", "; ")

        # 1) Count items
        counts_df = utilsbib.count_occurrences(
            self.df,
            item_col,
            count_type=count_type,
            item_column_name=item_column_name,
            translated_column_name=translated_column_name,
            sep=sep_for_count,
            **kwargs,
        )

        # Helper functions
        def _select_from_counts_by_regex(pattern: str) -> List[str]:
            mask = counts_df[item_column_name].astype(str).str.contains(
                pattern, regex=True, na=False
            )
            return counts_df.loc[mask, item_column_name].astype(str).tolist()

        def _dedup_preserve(seq):
            return list(dict.fromkeys(seq))

        # 2) Select items
        if items_included is not None:
            if isinstance(items_included, str):
                selected = _select_from_counts_by_regex(items_included)
            else:
                selected = [str(x) for x in items_included]
        else:
            if top_n and top_n > 0:
                selected = (
                    counts_df.nlargest(top_n, "Number of documents")[item_column_name]
                    .dropna().astype(str).tolist()
                )
            else:
                selected = (
                    counts_df.loc[
                        counts_df["Number of documents"] >= int(min_freq),
                        item_column_name
                    ]
                    .dropna().astype(str).tolist()
                )

        # Apply exclusion
        if items_excluded is not None and selected:
            if isinstance(items_excluded, str):
                pat = re.compile(items_excluded)
                selected = [it for it in selected if not pat.search(str(it))]
            else:
                excl_set = {str(x) for x in items_excluded}
                selected = [it for it in selected if str(it) not in excl_set]

        items_of_interest = _dedup_preserve(map(str, selected))

        # Edge case: nothing selected
        if len(items_of_interest) == 0:
            warnings.warn(
                f"No items selected for '{domain_key}' (min_freq={min_freq}, top_n={top_n}). "
                f"Total items counted: {len(counts_df)}. "
                f"Try lowering min_freq or using top_n parameter."
            )
            setattr(self, f"{domain_key}_associations", None)
            setattr(self, f"{domain_key}_contingency", pd.DataFrame())
            if filename:
                utilsbib._save_associations_xlsx(None, filename)
            return self

        # 3) Build indicators
        _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
            df=self.df,
            col=item_col,
            items_of_interest=items_of_interest,
            value_type=value_type,
        )
        binary_mat = indicators_dict["binary"]

        # 4) Relate groups to items
        G = (
            self.group_matrix.copy()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        X = (
            binary_mat.copy()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        common_idx = G.index.intersection(X.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No shared document index between group_matrix and the binary indicators."
            )
        G = G.loc[common_idx]
        X = X.loc[common_idx]

        relate_groups_to = getattr(self, "relate_groups_to", None)
        if callable(relate_groups_to):
            assoc, cont = relate_groups_to(
                domain_key,
                other_matrix=X,
                include_stats=include_stats,
                clean_zeros=clean_zeros,
                to_self=to_self,
            )
        else:
            assoc, cont = utilsbib.relate_concepts_general(
                "group",
                domain_key,
                known_matrices={"group": G},
                custom_matrices={domain_key: X},
                include_stats=include_stats,
                clean_zeros=clean_zeros,
                to_self=to_self,
            )

        # 5) Permutation inference (new in 2.16)
        if inference in ("permutation", "both"):
            self._attach_permutation_inference(
                assoc,
                G=G,
                X=X,
                tests=permutation_tests,
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                multiple_testing=multiple_testing,
                random_state=random_state,
            )

        # 6) Logit (new in 2.16)
        if logit != "none":
            self._attach_logit_results(
                assoc,
                G=G,
                X=X,
                method=logit,
                ci_method=logit_ci,
                compute_lrt=logit_compute_lrt,
            )

        setattr(self, f"{domain_key}_associations", assoc)
        setattr(self, f"{domain_key}_contingency", cont)

        # 7) Save if requested
        if filename:
            utilsbib._save_associations_xlsx(assoc, filename)

        return self

    # ------------------------------------------------------------------
    # New helpers (2.16)
    # ------------------------------------------------------------------

    def _attach_permutation_inference(
        self: BiblioGroup,
        relation: Any,
        *,
        G: pd.DataFrame,
        X: pd.DataFrame,
        tests: Tuple[str, ...],
        n_permutations: Optional[int],
        target_seconds: float,
        multiple_testing: str,
        random_state: Optional[int],
    ) -> None:
        """Run permutation tests and attach the results to ``relation``."""
        perm = _get_permutation_module()

        G_arr = G.to_numpy(dtype="float64")
        X_arr = X.to_numpy(dtype="float64")

        # Detect disjoint groups (asymptotic chi^2 already valid)
        disjoint = perm.is_partition(G_arr)
        relation.permutation_disjoint = disjoint
        if disjoint:
            warnings.warn(
                "Groups are disjoint: the asymptotic chi-squared test is "
                "unbiased; the permutation results below are reported for "
                "completeness only.",
                UserWarning, stacklevel=3,
            )

        # Global chi^2
        if "chi2" in tests:
            res = perm.assoc_permutation_test(
                G_arr, X_arr, test="chi2",
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                random_state=random_state,
                warn_disjoint=False,
            )
            relation.permutation_chi2 = float(res.observed)
            relation.permutation_chi2_p = float(res.p_value)
            relation.permutation_chi2_p_ci = res.p_value_ci
            relation.permutation_n = int(res.n_permutations)
            relation.permutation_seed = res.seed

        # Cramer's V
        if "cramers_v" in tests:
            res = perm.assoc_permutation_test(
                G_arr, X_arr, test="cramers_v",
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                random_state=random_state,
                warn_disjoint=False,
            )
            relation.permutation_cramers_v = float(res.observed)
            relation.permutation_cramers_v_p = float(res.p_value)

        # Total inertia (CA scalar)
        if "total_inertia" in tests:
            res = perm.assoc_permutation_test(
                G_arr, X_arr, test="total_inertia",
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                random_state=random_state,
                warn_disjoint=False,
            )
            relation.permutation_total_inertia = float(res.observed)
            relation.permutation_total_inertia_p = float(res.p_value)

        # Per-dimension CA inertias
        if "dimension_inertias" in tests:
            res = perm.assoc_permutation_test(
                G_arr, X_arr, test="dimension_inertias",
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                random_state=random_state,
                warn_disjoint=False,
            )
            relation.permutation_dim_inertias = np.asarray(res.observed)
            relation.permutation_dim_inertias_p = np.asarray(res.p_value)

        # Cell-level residuals
        if "residuals" in tests:
            res = perm.assoc_permutation_test(
                G_arr, X_arr, test="residuals",
                n_permutations=n_permutations,
                target_seconds=target_seconds,
                multiple_testing=multiple_testing,
                random_state=random_state,
                warn_disjoint=False,
            )
            relation.permutation_residuals = pd.DataFrame(
                np.asarray(res.observed), index=G.columns, columns=X.columns,
            )
            relation.permutation_residuals_p = pd.DataFrame(
                np.asarray(res.p_value), index=G.columns, columns=X.columns,
            )
            relation.permutation_residuals_p_adj = pd.DataFrame(
                res.extra["p_value_adjusted"],
                index=G.columns, columns=X.columns,
            )
            relation.permutation_residuals_correction = res.extra["multiple_testing"]

    def _attach_logit_results(
        self: BiblioGroup,
        relation: Any,
        *,
        G: pd.DataFrame,
        X: pd.DataFrame,
        method: str,
        ci_method: Optional[str],
        compute_lrt: bool,
    ) -> None:
        """Fit a per-group binary logit and attach results to ``relation``."""
        firth = _get_firth_module()

        results: Dict[str, Any] = {}
        method_used: Dict[str, str] = {}
        for g_name in G.columns:
            y = G[g_name].to_numpy(dtype="float64")
            if np.unique(y).size < 2:
                warnings.warn(
                    f"Skipping logit for group '{g_name}': "
                    f"only one class in the outcome.",
                    UserWarning, stacklevel=3,
                )
                continue
            try:
                res = firth.fit_logit(
                    X, y,
                    method=method,
                    ci_method=ci_method,
                    compute_lrt=compute_lrt,
                    add_intercept=True,
                )
                results[g_name] = res
                method_used[g_name] = res.method_used
            except Exception as exc:
                warnings.warn(
                    f"Logit fit failed for group '{g_name}': "
                    f"{type(exc).__name__}: {exc}",
                    UserWarning, stacklevel=3,
                )
        relation.logit_results = results
        relation.logit_method_used = method_used

    def _resolve_filename(
        self: BiblioGroup,
        filename: Optional[str],
        subfolder: str = "relations",
    ) -> Optional[str]:
        """Resolve filename with result folder path."""
        if filename is not None and getattr(self, "res_folder", None):
            return os.path.join(self.res_folder, subfolder, filename)
        return filename

    def associate_sources(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        filename: Optional[str] = "associated sources",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to journal sources.

        Parameters
        ----------
        include_stats : tuple
            Statistics to compute.
        clean_zeros : bool, default True
            Remove zero-count items.
        to_self : bool, default False
            Include self-associations.
        top_n : int, default 0
            Number of top items.
        min_freq : int, default 5
            Minimum frequency threshold.
        items_included, items_excluded : list or str, optional
            Filters for item selection.
        filename : str, optional
            Output filename.
        **kwargs :
            Additional options.

        Returns
        -------
        BiblioGroup
            Self for method chaining.
        """
        return self.associate_items(
            domain_key="sources",
            item_col="Source title",
            count_type="single",
            value_type="string",
            item_column_name="Source",
            translated_column_name="Abbreviated Source Title",
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_author_keywords(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated keywords",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to Author Keywords.

        Prefers processed keywords if available.
        """
        col = (
            "Processed Author Keywords"
            if "Processed Author Keywords" in self.df.columns
            else "Author Keywords"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="author_keywords",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Keyword",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_index_keywords(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated index keywords",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to Index Keywords.

        Prefers processed keywords if available.
        """
        col = (
            "Processed Index Keywords"
            if "Processed Index Keywords" in self.df.columns
            else "Index Keywords"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="index_keywords",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Keyword",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_abstract_words(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated abstract words",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to words from Abstracts.

        Prefers processed abstracts if available.
        """
        col = (
            "Processed Abstract"
            if "Processed Abstract" in self.df.columns
            else "Abstract"
        )

        extra = {}
        if sep is not None:
            extra["sep"] = sep

        return self.associate_items(
            domain_key="abstract_words",
            item_col=col,
            count_type="text",
            value_type="text",
            item_column_name="Word/Phrase",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **extra,
            **kwargs,
        )

    def associate_title_words(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated title words",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to words from Titles.

        Prefers processed titles if available.
        """
        col = (
            "Processed Title"
            if "Processed Title" in self.df.columns
            else "Title"
        )

        extra = {}
        if sep is not None:
            extra["sep"] = sep

        return self.associate_items(
            domain_key="title_words",
            item_col=col,
            count_type="text",
            value_type="list",
            item_column_name="Word/Phrase",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **extra,
            **kwargs,
        )

    def associate_authors(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated authors",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to Authors.

        Prefers full names if available.
        """
        col = (
            "Author full names"
            if "Author full names" in self.df.columns
            else "Authors"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="authors",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Author",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_countries(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated countries",
        **kwargs: Any,
    ) -> BiblioGroup:
        """
        Relate fixed groups to Countries.

        Uses 'CA Country' column if available, otherwise falls back to
        extracting countries from affiliations.
        """
        # Determine country column
        if "CA Country" in self.df.columns:
            col = "CA Country"
            count_type = "single"
            value_type = "string"
        elif "Countries of Authors" in self.df.columns:
            col = "Countries of Authors"
            count_type = "list"
            value_type = "list"
        else:
            # Try to create CA Country column
            self.df = utilsbib.add_ca_country_df(self.df, getattr(self, "db", ""))
            if "CA Country" in self.df.columns:
                col = "CA Country"
                count_type = "single"
                value_type = "string"
            else:
                raise KeyError(
                    "No country column found. Available columns with 'country' or 'affil': "
                    f"{[c for c in self.df.columns if 'country' in c.lower() or 'affil' in c.lower()]}"
                )

        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="countries",
            item_col=col,
            count_type=count_type,
            value_type=value_type,
            item_column_name="Country",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_affiliations(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated affiliations",
        **kwargs: Any,
    ) -> BiblioGroup:
        """Relate fixed groups to Affiliations."""
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="affiliations",
            item_col="Affiliations",
            count_type="list",
            value_type="list",
            item_column_name="Affiliation",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_references(
        self: BiblioGroup,
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated references",
        **kwargs: Any,
    ) -> BiblioGroup:
        """Relate fixed groups to References."""
        col = (
            "References"
            if "References" in self.df.columns
            else "Cited References"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="references",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Reference",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )
