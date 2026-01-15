# -*- coding: utf-8 -*-
"""
Extended classifier for predictive and statistical analysis of bibliometric
subgroup memberships, built on top of BiblioGroup.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from biblium.bibgroup import BiblioGroup
from typing import Sequence

class BiblioGroupClassifier(BiblioGroup):
    """
    Extended classifier for predictive and statistical analysis of
    bibliometric subgroup memberships.

    Inherits from :class:`BiblioGroup` and adds methods for:

    - Multi-model classification (scikit-learn).
    - Model evaluation and result export.
    - Logistic regression analysis (statsmodels).
    - Export of logistic regression results with significance highlighting.

    The class explicitly supports more than two, possibly overlapping
    subgroups by always treating each column of ``self.group_matrix`` as
    a separate binary dependent variable when performing logistic
    regression analysis, and by providing per-group and summary
    performance tables.

    Parameters
    ----------
    *args :
        Positional arguments forwarded to ``BiblioGroup.__init__``.
    features_columns : list of str, optional
        Column names in ``self.df`` to be used as numeric/tabular features.
        Used when no text columns are specified.
    text_columns : list of str, optional
        Column names in ``self.df`` that contain text to be vectorized
        with TF-IDF. If provided, text features are used instead of
        ``features_columns``.
    max_tfidf_features : int, default 500
        Maximum number of tokens for the TF-IDF vectorizer.
    max_count_features : int, default 500
        Maximum number of tokens for the CountVectorizer in the
        logistic regression analysis.
    **kwargs :
        Keyword arguments forwarded to ``BiblioGroup.__init__``.

    Attributes
    ----------
    features_columns : list of str
        Names of columns used as numeric features.
    text_columns : list of str
        Names of columns used as text features.
    max_tfidf_features : int
        Maximum number of TF-IDF tokens.
    max_count_features : int
        Maximum number of CountVectorizer tokens.
    features_df : pandas.DataFrame
        Cached numeric feature matrix (if used).
    vectorizer_ : TfidfVectorizer or None
        Fitted TF-IDF vectorizer when text features are used.
    """

    def __init__(
        self,
        *args,
        features_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        max_tfidf_features: int = 500,
        max_count_features: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.features_columns: List[str] = features_columns or []
        self.text_columns: List[str] = text_columns or []
        self.max_tfidf_features: int = max_tfidf_features
        self.max_count_features: int = max_count_features

        self.features_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer_: Optional[TfidfVectorizer] = None

    # ------------------------------------------------------------------
    # Feature preparation helpers
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        features_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Populate ``self.features_df`` by selecting columns from ``self.df``.

        Parameters
        ----------
        features_columns : list of str, optional
            If provided, overrides ``self.features_columns``.

        Returns
        -------
        pandas.DataFrame
            The numeric feature matrix.

        Raises
        ------
        ValueError
            If no feature columns are specified.
        KeyError
            If requested feature columns are not present in ``self.df``.
        """
        cols = features_columns or self.features_columns
        if not cols:
            raise ValueError(
                "No feature columns specified. "
                "Provide \"features_columns\" or set \"text_columns\"."
            )

        missing = set(cols) - set(self.df.columns)
        if missing:
            raise KeyError(f"Feature columns not found in df: {sorted(missing)}")

        self.features_df = self.df.loc[:, cols].copy()
        return self.features_df

    def _build_design_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        fit_vectorizer: bool = False,
    ) -> pd.DataFrame:
        """
        Build the design matrix (X) from either text or numeric features.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame to build features from. Defaults to ``self.df``.
        fit_vectorizer : bool, default False
            If True and text columns are used, fit a new TF-IDF vectorizer.
            If False, use an already-fitted vectorizer.

        Returns
        -------
        pandas.DataFrame
            Design matrix with samples in rows and features in columns.

        Raises
        ------
        ValueError
            If neither text nor numeric feature specifications are available.
        KeyError
            If requested numeric feature columns are not present in ``df``.
        """
        if df is None:
            df = self.df

        # Text-based features
        if self.text_columns:
            text_series = (
                df[self.text_columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
            )

            if fit_vectorizer or self.vectorizer_ is None:
                self.vectorizer_ = TfidfVectorizer(
                    max_features=self.max_tfidf_features
                )
                X_sparse = self.vectorizer_.fit_transform(text_series)
            else:
                X_sparse = self.vectorizer_.transform(text_series)

            return pd.DataFrame(
                X_sparse.toarray(),
                columns=self.vectorizer_.get_feature_names_out(),
                index=df.index,
            )

        # Numeric/tabular features
        if not self.features_columns:
            raise ValueError(
                "No numeric \"features_columns\" defined and no \"text_columns\" "
                "given; cannot build design matrix."
            )

        missing = set(self.features_columns) - set(df.columns)
        if missing:
            raise KeyError(
                f"Feature columns not found in provided DataFrame: {sorted(missing)}"
            )

        X = df.loc[:, self.features_columns].copy()
        if df is self.df:
            self.features_df = X.copy()
        return X

    # ------------------------------------------------------------------
    # Classification / evaluation
    # ------------------------------------------------------------------
    def evaluate_classifier(
        self,
        X,
        y,
        clf,
        method: str = "cross_validation",
        cv: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate a classifier using accuracy, AUC, precision, recall, and F1.

        Handles edge cases where the target has only one class by returning
        NaN metrics instead of raising an error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            Target labels. If 2D, treated as multilabel.
        clf : object
            Estimator implementing ``fit`` and ``predict`` (and ideally
            ``predict_proba`` or ``decision_function``).
        method : {"cross_validation", "leave_one_out", "train_test"}, \
                 default "cross_validation"
            Evaluation strategy.
        cv : int, default 5
            Number of folds when ``method="cross_validation"``.

        Returns
        -------
        dict
            Dictionary with keys ``{"accuracy", "roc_auc", "precision",
            "recall", "f1"}``. If evaluation is not possible (e.g. only one
            class present), all metrics are NaN.
        """
        metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
        results: Dict[str, float] = {m: float("nan") for m in metrics}

        y_arr = np.asarray(y)

        # 1) If y has only one class, skip cleanly
        if y_arr.ndim == 1:
            unique_classes = np.unique(y_arr)
            if unique_classes.size < 2:
                print("Skipping evaluation: target has only one class.")
                return results

        # 2) Cross-validation (binary only)
        if method in {"cross_validation", "leave_one_out"}:
            if y_arr.ndim != 1:
                # Multi-label or multi-output: fall back to train/test
                method = "train_test"
            else:
                cv_strategy = LeaveOneOut() if method == "leave_one_out" else cv
                try:
                    for m in metrics:
                        scores = cross_val_score(
                            clf,
                            X,
                            y_arr,
                            cv=cv_strategy,
                            scoring=m,
                        )
                        results[m] = float(np.mean(scores))
                except ValueError as exc:
                    # e.g. some CV split ends up with one class
                    print(f"Cross-validation failed: {exc}")
                return results

        # 3) Train/test split (binary or multilabel)
        stratify = y_arr if y_arr.ndim == 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_arr,
                test_size=0.2,
                random_state=0,
                stratify=stratify,
            )
        except ValueError as exc:
            # e.g. not enough samples per class for stratify
            print(f"train_test_split failed: {exc}")
            return results

        try:
            clf.fit(X_train, y_train)
        except ValueError as exc:
            # e.g. solver needs at least two classes in y
            print(f"Model fitting failed: {exc}")
            return results

        y_pred = clf.predict(X_test)

        # Accuracy, precision, recall, F1
        if y_arr.ndim == 1:
            results["accuracy"] = accuracy_score(y_test, y_pred)
            results["precision"] = precision_score(
                y_test, y_pred, zero_division=0
            )
            results["recall"] = recall_score(
                y_test, y_pred, zero_division=0
            )
            results["f1"] = f1_score(y_test, y_pred, zero_division=0)
        else:
            # Multilabel macro-averaged metrics
            results["accuracy"] = accuracy_score(y_test, y_pred)
            results["precision"] = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            results["recall"] = recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            results["f1"] = f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )

        # ROC AUC
        roc_auc = np.nan
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            proba = clf.decision_function(X_test)
        else:
            proba = None

        if proba is not None:
            try:
                if y_arr.ndim == 1:
                    # Binary
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        proba_1 = proba[:, 1]
                    else:
                        proba_1 = proba
                    roc_auc = roc_auc_score(y_test, proba_1)
                else:
                    # Multilabel macro AUC
                    roc_auc = roc_auc_score(
                        y_test, proba, average="macro"
                    )
            except ValueError:
                roc_auc = np.nan

        results["roc_auc"] = float(roc_auc)
        return results

    def classify_groups(
        self,
        classifiers: Optional[Dict[str, object]] = None,
        method: str = "cross_validation",
        multilabel: bool = False,
        save_results: bool = False,
        file_prefix: str = "results",
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Classify each group (or multilabel target) with multiple models.

        Dependent variables are always taken from ``self.group_matrix`` columns.

        In the usual (non-multilabel) mode, this method treats each column
        of ``self.group_matrix`` as a separate binary problem and evaluates
        each classifier on each group. This naturally supports more than two,
        possibly overlapping, subgroups.

        Parameters
        ----------
        classifiers : dict, optional
            Mapping ``name -> estimator``. If None, a default set is used:
            LogisticRegression, RandomForest, GradientBoosting,
            MultinomialNB, SVC(probability=True).
        method : {"cross_validation", "leave_one_out", "train_test"}, \
                 default "cross_validation"
            Evaluation strategy for each model.
        multilabel : bool, default False
            If True, treat ``self.group_matrix`` as a multilabel target and
            fit a single One-vs-Rest classifier per model.
        save_results : bool, default False
            If True, save the performance table and summaries to an Excel file.
        file_prefix : str, default "results"
            Prefix for the Excel filename (``<prefix>.xlsx``).

        Returns
        -------
        dict
            If ``multilabel=False``:
                ``{group_name: {model_name: metrics_dict}}``
            If ``multilabel=True``:
                ``{"multilabel": {model_name: metrics_dict}}``
        """
        if classifiers is None:
            classifiers = {
                "Logistic": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "GBM": GradientBoostingClassifier(),
                "NaiveBayes": MultinomialNB(),
                "SVM": SVC(probability=True),
            }

        # Build feature matrix on current data
        X = self._build_design_matrix(fit_vectorizer=True)
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        if multilabel:
            # Multi-label: joint target from all columns of self.group_matrix
            y = self.group_matrix.values
            multilabel_results: Dict[str, Dict[str, float]] = {}

            for name, clf in classifiers.items():
                ovr = OneVsRestClassifier(clf)
                metrics = self.evaluate_classifier(
                    X, y, ovr, method="train_test"
                )
                multilabel_results[name] = metrics

            results["multilabel"] = multilabel_results
        else:
            # One binary problem per group
            for grp in self.group_matrix.columns:
                y = self.group_matrix[grp].values
                group_results: Dict[str, Dict[str, float]] = {}
                for name, clf in classifiers.items():
                    metrics = self.evaluate_classifier(X, y, clf, method=method)
                    group_results[name] = metrics
                results[grp] = group_results

        if save_results:
            self._save_performance(results, f"{file_prefix}.xlsx")

        return results

    def _save_performance(self, results: Dict[str, dict], fname: str) -> None:
        """
        Save classification performance results to an Excel file.

        The main sheet ``Performance`` contains one row per (group, model) with
        all computed metrics. If a ``\"multilabel\"`` entry is present in
        ``results``, it is stored as a pseudo-group named ``\"multilabel\"``.

        Two additional summary sheets are created:

        - ``Summary_by_Model`` – average, standard deviation, minimum, and
          maximum of each metric across groups, for every model.
        - ``Summary_by_Group`` – the same summary statistics across models,
          for every group.

        Parameters
        ----------
        results : dict
            Output of :meth:`classify_groups`.
        fname : str
            Excel filename.
        """
        rows: List[Dict[str, Union[str, float]]] = []

        # Multilabel block (if present)
        if "multilabel" in results:
            for model_name, metrics in results["multilabel"].items():
                rows.append(
                    {
                        "Group": "multilabel",
                        "Model": model_name,
                        **metrics,
                    }
                )
            group_items = {k: v for k, v in results.items() if k != "multilabel"}
        else:
            group_items = results

        # Per-group blocks
        for grp, models in group_items.items():
            for model_name, metrics in models.items():
                rows.append(
                    {
                        "Group": grp,
                        "Model": model_name,
                        **metrics,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            # Nothing to save – create an empty workbook with a placeholder sheet.
            with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
                pd.DataFrame(
                    {"Message": ["No performance results available."]}
                ).to_excel(writer, sheet_name="Performance", index=False)
            return

        metric_cols = [c for c in df.columns if c not in {"Group", "Model"}]

        with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
            # Main table
            df.to_excel(writer, sheet_name="Performance", index=False)

            if metric_cols:
                # Summary by model
                summary_by_model = (
                    df.groupby("Model")[metric_cols]
                    .agg(["mean", "std", "min", "max"])
                    .sort_index()
                )
                summary_by_model.to_excel(
                    writer, sheet_name="Summary_by_Model"
                )

                # Summary by group
                summary_by_group = (
                    df.groupby("Group")[metric_cols]
                    .agg(["mean", "std", "min", "max"])
                    .sort_index()
                )
                summary_by_group.to_excel(
                    writer, sheet_name="Summary_by_Group"
                )

    def train_classifier(
        self,
        clf,
        multilabel: bool = False,
    ) -> Union[
        Callable[[pd.DataFrame], np.ndarray],
        Dict[str, Callable[[pd.DataFrame], np.ndarray]],
    ]:
        """
        Train a classifier and return convenient prediction functions.

        Dependent variables are always taken from ``self.group_matrix`` columns.

        If a group has only one class (all 0 or all 1), a constant predictor
        is used for that group instead of fitting a model.

        Parameters
        ----------
        clf : estimator object
            Any scikit-learn compatible estimator implementing ``fit`` and
            ``predict`` (and ideally ``predict_proba``).
        multilabel : bool, default False
            If True, fit a single One-vs-Rest classifier for all groups and
            return a single prediction function that outputs a 2D array
            (shape: ``n_samples x n_groups``). If False, fit one classifier
            per group and return a dict of prediction functions.

        Returns
        -------
        callable or dict
            If ``multilabel=True``:
                ``predict(new_df) -> np.ndarray`` (2D).
            If ``multilabel=False``:
                ``{group_name: predict_fn}``
                where ``predict_fn(new_df) -> np.ndarray`` is 1D.

        Raises
        ------
        ValueError
            If ``multilabel=True`` but no group has at least two classes.
        """
        # Fit design matrix on the full current data
        X = self._build_design_matrix(fit_vectorizer=True)

        # ---------- multilabel case ----------
        if multilabel:
            gm = self.group_matrix.copy()

            good_groups = [g for g in gm.columns if gm[g].nunique() >= 2]
            const_groups = [g for g in gm.columns if gm[g].nunique() < 2]

            if not good_groups:
                raise ValueError(
                    "No groups with at least two classes; cannot train multilabel classifier."
                )

            y_good = gm[good_groups].values
            ovr = OneVsRestClassifier(clf)
            ovr.fit(X, y_good)

            const_vals = {g: gm[g].iloc[0] for g in const_groups}
            all_groups = list(gm.columns)
            idx_map = {g: i for i, g in enumerate(all_groups)}

            def predict_multilabel(new_df: pd.DataFrame) -> np.ndarray:
                """
                Predict multilabel group memberships for new data.

                Parameters
                ----------
                new_df : pandas.DataFrame
                    New documents for prediction.

                Returns
                -------
                numpy.ndarray
                    Array of shape ``(n_samples, n_groups)`` with predicted
                    binary memberships, ordered according to the columns of
                    ``self.group_matrix``.
                """
                X_new = self._build_design_matrix(
                    df=new_df, fit_vectorizer=False
                )
                y_good_pred = ovr.predict(X_new)  # (n_samples, len(good_groups))

                n = len(new_df)
                n_groups = len(all_groups)
                out = np.zeros((n, n_groups), dtype=y_good_pred.dtype)

                # Fill good groups
                for j, g in enumerate(good_groups):
                    out[:, idx_map[g]] = y_good_pred[:, j]

                # Fill constant groups
                for g, v in const_vals.items():
                    out[:, idx_map[g]] = v

                return out

            return predict_multilabel

        # ---------- per-group case ----------
        predictors: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {}

        for grp in self.group_matrix.columns:
            y = self.group_matrix[grp].values
            unique = np.unique(y)

            # Only one class -> constant predictor
            if unique.size < 2:
                const_val = unique[0]
                print(
                    f"Skipping training for group \"{grp}\": "
                    f"only one class ({const_val}) present. Using constant predictor."
                )

                def const_predict_fn(
                    new_df: pd.DataFrame,
                    v=const_val,
                ) -> np.ndarray:
                    """
                    Constant prediction function used when a group has
                    only one observed class.

                    Parameters
                    ----------
                    new_df : pandas.DataFrame
                        New documents for prediction.
                    v :
                        Constant value to predict for all samples.

                    Returns
                    -------
                    numpy.ndarray
                        One-dimensional array of constant predictions.
                    """
                    return np.full(
                        shape=(len(new_df),),
                        fill_value=v,
                        dtype=type(v),
                    )

                predictors[grp] = const_predict_fn
                continue

            # Normal case
            model = clone(clf)
            model.fit(X, y)

            def make_predict_fn(
                model_: object,
            ) -> Callable[[pd.DataFrame], np.ndarray]:
                """
                Wrap a fitted model in a convenient prediction function.

                Parameters
                ----------
                model_ : estimator
                    Fitted scikit-learn estimator.

                Returns
                -------
                callable
                    Function ``predict_fn(new_df) -> np.ndarray`` producing
                    one-dimensional predictions for the given group.
                """

                def predict_fn(new_df: pd.DataFrame) -> np.ndarray:
                    X_new = self._build_design_matrix(
                        df=new_df, fit_vectorizer=False
                    )
                    return model_.predict(X_new)

                return predict_fn

            predictors[grp] = make_predict_fn(model)

        return predictors

    # ------------------------------------------------------------------
    # Logistic regression analysis (statsmodels)
    # ------------------------------------------------------------------
    

    def logistic_regression_analysis(
        self,
        text_column: str = "Abstract",
        include_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
        top_n: int = 50,
        groups: Optional[List[str]] = None,
        save_to: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        items_of_interest: Optional[Sequence[str]] = None,
    ) -> Dict[str, dict]:
        """
        Perform separate logistic regressions for each subgroup in ``group_matrix``.
    
        Each column in ``self.group_matrix`` is treated as a binary dependent
        variable (membership vs. non-membership). This naturally supports
        settings with more than two, possibly overlapping, groups.
    
        Design matrix of independent variables:
    
        - If ``X`` is None (text mode):
            * Build a binary term matrix from ``text_column`` using
              ``CountVectorizer``.
            * If the column name contains ``"keyword"`` (case-insensitive),
              it is treated as a keyword list: cells are split on
              ``self.default_separator`` and entire keywords are used as
              tokens (multi-word phrases preserved).
            * Otherwise, standard word tokenization with English stop words.
            * Terms are filtered via ``include_regex`` / ``exclude_regex``.
            * If ``items_of_interest`` is provided, only those terms that
              remain after filtering are used and ``top_n`` is ignored.
            * Otherwise, keep the ``top_n`` most frequent terms
              (or all when ``top_n <= 0``).
    
        - If ``X`` is provided (matrix mode):
            * ``X`` can be a DataFrame or ndarray.
            * Column names (or generated names x0, x1, ...) are used as
              predictor labels and filtered via ``include_regex`` /
              ``exclude_regex``.
            * If ``items_of_interest`` is provided, only those predictors
              that remain after filtering are used and ``top_n`` is ignored.
            * Otherwise, keep the ``top_n`` predictors with the highest
              non-zero document counts (or all when ``top_n <= 0``).
    
        In both modes, a constant term is added via
        ``statsmodels.api.add_constant``.
    
        Parameters
        ----------
        text_column : str, default "Abstract"
            Name of the column in ``self.df`` containing text or keyword lists.
            Only used when ``X`` is None.
        include_regex : str, optional
            Keep only terms / predictors whose string representation matches
            this regex.
        exclude_regex : str, optional
            Drop terms / predictors whose string representation matches this
            regex.
        top_n : int, default 50
            Maximum number of predictors to keep after filtering. If
            ``top_n <= 0``, all remaining predictors are used. Ignored when
            ``items_of_interest`` is provided.
        groups : list of str, optional
            Optional subset of column names from ``self.group_matrix`` to use
            as dependent variables. If None, all columns are used.
        save_to : str, optional
            If provided, the results are exported to this Excel file via
            :meth:`save_logistic_results` and the same results dict is returned.
        X : pandas.DataFrame or numpy.ndarray, optional
            User-provided design matrix. If None, a term matrix is built from
            ``text_column`` as described above.
        items_of_interest : sequence of str, optional
            Optional list of predictor names (terms or column names) to be
            used as independent variables. Only those present after filtering
            are kept. If provided, ``top_n`` is ignored.
    
        Returns
        -------
        dict
            Mapping
            ``group_name -> {"model": LogitResults, "summary": pandas.DataFrame}``.
            Groups with constant outcomes are skipped with a warning.
    
        Raises
        ------
        KeyError
            If required columns in ``self.df`` or requested groups are missing.
        ValueError
            If no predictors remain after filtering.
        """
        # ------------------------------------------------------------------
        # Dependent variables: select columns from self.group_matrix
        # ------------------------------------------------------------------
        if self.group_matrix is None:
            raise KeyError(
                "self.group_matrix is None. Set it before calling logistic_regression_analysis()."
            )
    
        if groups is None:
            groups = list(self.group_matrix.columns)
        else:
            missing = set(groups) - set(self.group_matrix.columns)
            if missing:
                raise KeyError(
                    f"Groups not found in self.group_matrix: {sorted(missing)}"
                )
    
        # ------------------------------------------------------------------
        # Build design matrix X_design and items_df
        # ------------------------------------------------------------------
        if X is None:
            # ---- Text-based mode (with special handling for keyword columns) ----
            if text_column not in self.df.columns:
                raise KeyError(f'Column "{text_column}" not found in df.')
    
            texts = self.df[text_column].fillna("").astype(str)
    
            is_keyword_col = "keyword" in text_column.lower()
    
            if is_keyword_col:
                # Treat cell as list of keywords separated by self.default_separator
                sep = getattr(self, "default_separator", "; ")
    
                def keyword_tokenizer(text: str, _sep=sep) -> List[str]:
                    if not text:
                        return []
                    return [
                        t.strip()
                        for t in str(text).split(_sep)
                        if t.strip()
                    ]
    
                vec = CountVectorizer(
                    tokenizer=keyword_tokenizer,
                    token_pattern=None,
                    max_features=self.max_count_features,
                    binary=True,
                )
            else:
                # Standard word-based tokenization with English stop words
                vec = CountVectorizer(
                    max_features=self.max_count_features,
                    stop_words="english",
                    binary=True,
                )
    
            Xc = vec.fit_transform(texts)
    
            vocab = np.array(vec.get_feature_names_out())
            doc_counts = (Xc > 0).sum(axis=0).A1
    
            items_df = pd.DataFrame({"item": vocab, "doc_count": doc_counts})
    
            # Regex filters on term strings
            if include_regex:
                items_df = items_df[
                    items_df["item"].astype(str).str.contains(include_regex, regex=True)
                ]
            if exclude_regex:
                items_df = items_df[
                    ~items_df["item"].astype(str).str.contains(exclude_regex, regex=True)
                ]
    
            # Determine which items to use
            if items_of_interest is not None:
                items_set = set(items_df["item"])
                selected_items = [it for it in items_of_interest if it in items_set]
                if not selected_items:
                    raise ValueError(
                        "No items_of_interest found in the vocabulary after filtering."
                    )
                items = selected_items
            else:
                items_df = items_df.sort_values("doc_count", ascending=False)
                if top_n > 0:
                    items = items_df.head(top_n)["item"].tolist()
                else:
                    items = items_df["item"].tolist()
    
            if not items:
                raise ValueError("No predictors remain after filtering (text mode).")
    
            X_dense = Xc.toarray()
            X_all = pd.DataFrame(
                X_dense,
                columns=vocab,
                index=self.df.index,
            )
            X_design = X_all[items]
    
        else:
            # ---- User-provided X mode ----
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
                # Align index if necessary
                if not X_df.index.equals(self.df.index):
                    if len(X_df) != len(self.df):
                        raise ValueError(
                            "Provided X has different number of rows than self.df."
                        )
                    X_df.index = self.df.index
            else:
                X_arr = np.asarray(X)
                if X_arr.shape[0] != len(self.df):
                    raise ValueError(
                        "Provided X has different number of rows than self.df."
                    )
                col_names = [f"x{i}" for i in range(X_arr.shape[1])]
                X_df = pd.DataFrame(X_arr, index=self.df.index, columns=col_names)
    
            # Drop any existing constant column named "const"
            predictor_cols = [c for c in X_df.columns if c != "const"]
            X_pred = X_df[predictor_cols]
    
            # Document counts as number of non-zero entries per predictor
            doc_counts = (X_pred != 0).sum(axis=0).astype(int).to_numpy()
            items_df = pd.DataFrame(
                {"item": X_pred.columns.astype(str), "doc_count": doc_counts}
            )
    
            # Regex filters on column names
            if include_regex:
                items_df = items_df[
                    items_df["item"].astype(str).str.contains(include_regex, regex=True)
                ]
            if exclude_regex:
                items_df = items_df[
                    ~items_df["item"].astype(str).str.contains(exclude_regex, regex=True)
                ]
    
            # Determine predictors to use
            if items_of_interest is not None:
                items_set = set(items_df["item"])
                selected_items = [it for it in items_of_interest if it in items_set]
                if not selected_items:
                    raise ValueError(
                        "No items_of_interest found among X columns after filtering."
                    )
                items = selected_items
            else:
                items_df = items_df.sort_values("doc_count", ascending=False)
                if top_n > 0:
                    items = items_df.head(top_n)["item"].tolist()
                else:
                    items = items_df["item"].tolist()
    
            if not items:
                raise ValueError("No predictors remain after filtering (X mode).")
    
            X_design = X_pred[items]
    
        # Add intercept
        X_design = sm.add_constant(X_design, has_constant="add")
    
        # ------------------------------------------------------------------
        # Fit one binary logit per group
        # ------------------------------------------------------------------
        results: Dict[str, dict] = {}
    
        for grp in groups:
            y = self.group_matrix[grp].values
    
            # Skip groups with less than two classes
            if np.unique(y).size < 2:
                print(
                    f'Skipping logistic regression for group "{grp}": '
                    "dependent variable has only one class."
                )
                continue
    
            try:
                model = sm.Logit(y, X_design).fit(disp=False)
                coef_table = model.summary2().tables[1]
                results[grp] = {"model": model, "summary": coef_table}
            except Exception as exc:  # singular matrix, separation, etc.
                print(f'Logit failed for group "{grp}": {exc}')
    
        # Optional: compute + save in one call
        if save_to is not None and results:
            self.save_logistic_results(results, filename=save_to)
    
        return results


    def save_logistic_results(
        self,
        results: Dict[str, dict],
        filename: str = "logistic_results.xlsx",
    ) -> None:
        """
        Export logistic regression results to Excel with p-value highlighting,
        direction arrows, and group-level summaries.
    
        For each group, two sheets are created:
    
        - "coefficients <group>": coefficient table + odds ratios, plus a
          "Direction" column:
              * "↑"   / "↓"   for 0.01 < p <= 0.05 (positive / negative)
              * "↑↑"  / "↓↓"  for 0.001 < p <= 0.01
              * "↑↑↑" / "↓↓↓" for p <= 0.001
              * ""           otherwise (including the intercept)
          P-values are color-coded by significance.
    
        - "statistics <group>": model-level statistics (AIC, BIC, pseudo R²).
    
        A combined sheet "Combined_Coefficients" concatenates per-group
        coefficients (Coef., OR, P>|z|, Direction) horizontally.
    
        A "Summary" sheet provides one row per group with:
    
        - number of observations,
        - number of predictors (excluding the constant),
        - counts of significant terms at p ≤ 0.10, 0.05, 0.01, 0.001,
        - AIC, BIC, pseudo R².
    
        Parameters
        ----------
        results : dict
            Output from :meth:`logistic_regression_analysis`.
        filename : str, default "logistic_results.xlsx"
            Excel filename.
    
        Raises
        ------
        ValueError
            If ``results`` is empty.
        """
        if not results:
            raise ValueError("No logistic regression results to save.")
    
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            workbook = writer.book
    
            # p-value formats
            p_formats = {
                0.001: workbook.add_format(
                    {"bg_color": "#006400", "font_color": "#FFFFFF"}
                ),
                0.01: workbook.add_format(
                    {"bg_color": "#228B22", "font_color": "#FFFFFF"}
                ),
                0.05: workbook.add_format(
                    {"bg_color": "#66CDAA", "font_color": "#000000"}
                ),
                0.1: workbook.add_format(
                    {"bg_color": "#98FB98", "font_color": "#000000"}
                ),
            }
    
            combined_tables: List[pd.DataFrame] = []
            summary_rows: List[Dict[str, Union[str, float, int]]] = []
    
            for grp, data in results.items():
                coef_df = data["summary"].copy()
                model = data["model"]
    
                # Odds ratios
                coef_df["OR"] = np.exp(coef_df["Coef."])
    
                # ---------------------------------------------------------
                # Direction arrows based on sign and p-value
                # ---------------------------------------------------------
                if "Coef." in coef_df.columns and "P>|z|" in coef_df.columns:
                    coef_vals = pd.to_numeric(coef_df["Coef."], errors="coerce")
                    p_vals = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
    
                    direction = np.full(len(coef_df), "", dtype=object)
    
                    # Sign masks
                    pos = coef_vals > 0
                    neg = coef_vals < 0
    
                    # p-value bands
                    band1 = (p_vals <= 0.05) & (p_vals > 0.01)
                    band2 = (p_vals <= 0.01) & (p_vals > 0.001)
                    band3 = p_vals <= 0.001
    
                    # Assign arrows
                    direction[band1 & pos] = "↑"
                    direction[band1 & neg] = "↓"
    
                    direction[band2 & pos] = "↑↑"
                    direction[band2 & neg] = "↓↓"
    
                    direction[band3 & pos] = "↑↑↑"
                    direction[band3 & neg] = "↓↓↓"
    
                    # No arrows for the intercept
                    if "const" in coef_df.index:
                        direction[coef_df.index == "const"] = ""
    
                    coef_df["Direction"] = direction
                else:
                    coef_df["Direction"] = ""
    
                # ---------------------------------------------------------
                # Coefficients sheet
                # ---------------------------------------------------------
                sheet_name_coef = f"coeff {grp}"
                coef_df.to_excel(writer, sheet_name=sheet_name_coef[:32], index=True)
                ws_coef = writer.sheets[sheet_name_coef]
    
                # Highlight p-values in the per-group sheet
                if "P>|z|" in coef_df.columns:
                    p_col_idx = coef_df.columns.get_loc("P>|z|") + 1  # +1 for index
                    # For a normal DataFrame, data rows start at row=1 (0-based)
                    for row_idx, p_val in enumerate(coef_df["P>|z|"], start=1):
                        if pd.isna(p_val):
                            continue
                        for threshold, fmt in p_formats.items():
                            if p_val <= threshold:
                                ws_coef.write(row_idx, p_col_idx, p_val, fmt)
                                break
    
                # ---------------------------------------------------------
                # Statistics sheet
                # ---------------------------------------------------------
                stats_df = pd.DataFrame(
                    {
                        "AIC": [model.aic],
                        "BIC": [model.bic],
                        "Pseudo R-squared": [model.prsquared],
                    }
                )
                stats_df.to_excel(
                    writer, sheet_name=f"stats {grp}"[:32], index=False
                )
    
                # ---------------------------------------------------------
                # Data for combined coefficients (include Direction)
                # ---------------------------------------------------------
                comb_df = coef_df[["Coef.", "OR", "P>|z|", "Direction"]].copy()
                comb_df.columns = pd.MultiIndex.from_product(
                    [[grp], comb_df.columns]
                )
                combined_tables.append(comb_df)
    
                # ---------------------------------------------------------
                # Row for summary sheet
                # ---------------------------------------------------------
                coef_no_const = coef_df.drop(index="const", errors="ignore")
                if "P>|z|" in coef_no_const.columns:
                    pvals = coef_no_const["P>|z|"].dropna()
                else:
                    pvals = pd.Series([], dtype=float)
    
                summary_rows.append(
                    {
                        "Group": grp,
                        "N_obs": int(model.nobs),
                        "N_terms": int(coef_no_const.shape[0]),
                        "Sig(p<=0.10)": int((pvals <= 0.10).sum()),
                        "Sig(p<=0.05)": int((pvals <= 0.05).sum()),
                        "Sig(p<=0.01)": int((pvals <= 0.01).sum()),
                        "Sig(p<=0.001)": int((pvals <= 0.001).sum()),
                        "AIC": model.aic,
                        "BIC": model.bic,
                        "Pseudo R-squared": model.prsquared,
                    }
                )
    
            # -------------------------------------------------------------
            # Combined coefficients across groups
            # -------------------------------------------------------------
            if combined_tables:
                combined = pd.concat(combined_tables, axis=1).dropna(how="all")
                combined.to_excel(
                    writer, sheet_name="Combined_Coefficients", index=True
                )
                ws_comb = writer.sheets["Combined_Coefficients"]
    
                # For MultiIndex columns, pandas writes:
                #   - one row per column level
                #   - plus a blank row
                # Data therefore start at row = nlevels + 2 (1-based),
                # i.e. row index (0-based) = nlevels + 1.
                if isinstance(combined.columns, pd.MultiIndex):
                    start_row = combined.columns.nlevels + 1
                else:
                    start_row = 1
    
                # Highlight p-values in combined sheet
                for grp in results:
                    col_key = (grp, "P>|z|") if isinstance(
                        combined.columns, pd.MultiIndex
                    ) else "P>|z|"
                    if col_key not in combined.columns:
                        continue
    
                    col_idx = combined.columns.get_loc(col_key) + 1  # +1 for index
    
                    for row_idx, p_val in enumerate(
                        combined[col_key], start=start_row
                    ):
                        if pd.isna(p_val):
                            continue
                        for threshold, fmt in p_formats.items():
                            if p_val <= threshold:
                                ws_comb.write(row_idx, col_idx, p_val, fmt)
                                break
    
            # -------------------------------------------------------------
            # Summary sheet (one row per group)
            # -------------------------------------------------------------
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
