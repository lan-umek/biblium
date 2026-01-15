# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:21:17 2025

@author: Lan
"""

from biblium.bibstats import BiblioStats
from biblium.bibgroup import BiblioGroup
from biblium import plotbib, utilsbib
import numpy as np
import re
import os
import logging
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# Try to import co_mapping, fallback if not available
try:
    from mappingbib import co_mapping
except ImportError:
    co_mapping = {}

class BiblioPlot(BiblioStats):
    
    
    def plot_average_citations_per_year(self, filename_base="average citations per document", **kwargs):
        """Plot average citations per document by publication year.

        This is a thin wrapper around ``utilsbib.compute_average_citations_per_year``
        and ``plotbib.plot_average_citations_per_year`` that uses the instance data
        frame and result folder.

        Parameters
        ----------
        filename_base : str or None, default "average citations per document"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``. If None, the plot is not saved.
        **kwargs :
            Additional keyword arguments forwarded to
            ``plotbib.plot_average_citations_per_year``.
        """
        grouped = utilsbib.compute_average_citations_per_year(self.df)
        if filename_base is not None:
            filename_base = os.path.join(self.res_folder, "plots", filename_base)
            plotbib.plot_average_citations_per_year(grouped, **kwargs)

    
    def dist_plots(
        self,
        grouping_vars=["Source title", "Document Type"],
        numeric_vars=["Year", "Cited by", "Field-Weighted Citation Impact",
                      "Citation Normalized Percentile"],
        list_grouping_vars=None,
        max_groups=5,
        order_by_size=True,
        plot_type="box",
        **kwargs,
    ):
        """
        Generate box or violin plots for combinations of numeric and grouping variables.
    
        Parameters
        ----------
        grouping_vars : list of str
            Categorical variables to use for grouping (single value per row).
        numeric_vars : list of str
            Numerical variables to be plotted.
        list_grouping_vars : list of str, optional
            Variables where each cell contains a delimited list of items
            separated by ``self.default_separator``. The entries
            "Author Keywords" and "Index Keywords" will prefer the processed
            versions "Processed Author Keywords" and "Processed Index Keywords"
            if those columns are available. The list can be empty.
        max_groups : int
            Maximum number of groups to show per plot.
        order_by_size : bool
            Whether to order groups by size.
        plot_type : str
            Either "box" or "violin".
        **kwargs : dict
            Additional keyword arguments passed to the plot function.
        """
        import os
    
        plot_func = {
            "box": plotbib.plot_boxplot,
            "violin": plotbib.plot_violinplot,
        }.get(plot_type)
    
        if plot_func is None:
            raise ValueError("plot_type must be 'box' or 'violin'")
    
        if list_grouping_vars is None:
            list_grouping_vars = []
    
        # Keep only existing columns for simple grouping / numeric vars
        grouping_vars = [c for c in grouping_vars if c in self.df.columns]
        numeric_vars = [c for c in numeric_vars if c in self.df.columns]
    
        def _explode_list_column(df, col, sep):
            """Explode a delimited string column into multiple rows."""
            tmp = df.dropna(subset=[col]).copy()
            tmp[col] = tmp[col].astype(str).str.split(sep)
            tmp = tmp.explode(col)
            tmp[col] = tmp[col].str.strip()
            tmp = tmp[tmp[col] != ""]
            return tmp
    
        # --------- Plots for standard (single-valued) grouping vars ----------
        for group_var in grouping_vars:
            for numeric_var in numeric_vars:
                filename_base = os.path.join(
                    self.res_folder,
                    "plots",
                    f"{numeric_var} by {group_var}_{plot_type}",
                )
                plot_func(
                    self.df,
                    value_column=numeric_var,
                    group_by=group_var,
                    max_groups=max_groups,
                    order_by_size=order_by_size,
                    filename_base=filename_base,
                    dpi=self.dpi,
                    **kwargs,
                )
    
        # --------- Plots for list-like (multi-valued) grouping vars ----------
        resolved_list_grouping = []
    
        for gv in list_grouping_vars:
            actual_col = None
            display_name = gv
    
            # Prefer processed author keywords
            if gv in ("Author Keywords", "Authors Keywords", "author keywords"):
                if "Processed Author Keywords" in self.df.columns:
                    actual_col = "Processed Author Keywords"
                    display_name = "Author Keywords"
                elif "Author Keywords" in self.df.columns:
                    actual_col = "Author Keywords"
                    display_name = "Author Keywords"
                elif "Authors Keywords" in self.df.columns:
                    actual_col = "Authors Keywords"
                    display_name = "Author Keywords"
    
            # Prefer processed index keywords
            elif gv in ("Index Keywords", "index keywords"):
                if "Processed Index Keywords" in self.df.columns:
                    actual_col = "Processed Index Keywords"
                    display_name = "Index Keywords"
                elif "Index Keywords" in self.df.columns:
                    actual_col = "Index Keywords"
                    display_name = "Index Keywords"
    
            # Generic list-like column
            else:
                if gv in self.df.columns:
                    actual_col = gv
                    display_name = gv
    
            if actual_col is not None:
                resolved_list_grouping.append((display_name, actual_col))
    
        for display_name, actual_col in resolved_list_grouping:
            df_exploded = _explode_list_column(
                self.df,
                actual_col,
                self.default_separator,
            )
            if display_name != actual_col:
                df_exploded = df_exploded.rename(columns={actual_col: display_name})
    
            for numeric_var in numeric_vars:
                filename_base = os.path.join(
                    self.res_folder,
                    "plots",
                    f"{numeric_var} by {display_name}_{plot_type}",
                )
                plot_func(
                    df_exploded,
                    value_column=numeric_var,
                    group_by=display_name,
                    max_groups=max_groups,
                    order_by_size=order_by_size,
                    filename_base=filename_base,
                    dpi=self.dpi,
                    **kwargs,
                )

                
    
    def plot_scientific_production(self, filename="scientific production", **kwargs):
        """Plot annual scientific production based on precomputed statistics.

        If ``production_df`` is not yet available it is computed via
        :meth:`get_production`. The result is then passed to
        :func:`plotbib.plot_timeseries`.

        Parameters
        ----------
        filename : str, default "scientific production"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_timeseries`.
        """
        filename = os.path.join(self.res_folder, "plots", filename)
        if not hasattr(self, "production_df"):
            self.get_production()
        plotbib.plot_timeseries(self.production_df, filename=filename, dpi=self.dpi, **kwargs)

    def plot_growth_model(
        self, 
        model_type: str = "auto",
        forecast_years: int = 5,
        filename: str = "growth_model",
        show_residuals: bool = False,
        **kwargs
    ):
        """
        Fit and plot bibliometric growth models.
        
        Fits exponential, logistic, power law, or linear models to annual
        publication counts and creates visualization with forecast.
        
        Parameters
        ----------
        model_type : str
            Model type: "exponential", "logistic", "power", "linear", "auto".
        forecast_years : int
            Number of years to forecast.
        filename : str
            Base filename for saving.
        show_residuals : bool
            If True, includes residuals analysis plots.
        **kwargs :
            Additional arguments passed to plotbib.plot_growth_model().
        
        Returns
        -------
        dict
            Growth model results including predictions.
        
        Examples
        --------
        >>> bib.plot_growth_model(model_type="auto", forecast_years=10)
        >>> bib.plot_growth_model(model_type="logistic", show_residuals=True)
        """
        # Fit model if not already done
        result = self.fit_growth_model(
            model_type=model_type,
            forecast_years=forecast_years,
            verbose=False
        )
        
        # Plot
        if filename is not None:
            filename = os.path.join(self.res_folder, "plots", filename)
        
        plotbib.plot_growth_model(
            result,
            filename=filename,
            dpi=self.dpi,
            show_residuals=show_residuals,
            **kwargs
        )
        
        return result

    def plot_life_cycle(
        self,
        forecast_years: int = 10,
        filename: str = "life_cycle",
        **kwargs
    ):
        """
        Fit and plot life cycle (S-curve) analysis.
        
        Analyzes the maturity of a research field by fitting a logistic
        growth model to cumulative publications.
        
        Parameters
        ----------
        forecast_years : int
            Years to forecast.
        filename : str
            Base filename for saving.
        **kwargs :
            Additional arguments passed to plotbib.plot_life_cycle().
        
        Returns
        -------
        dict
            Life cycle analysis results.
        
        Examples
        --------
        >>> result = bib.plot_life_cycle(forecast_years=15)
        >>> print(f"Current phase: {result['current_phase']}")
        """
        # Fit model
        result = self.fit_life_cycle_model(
            forecast_years=forecast_years,
            verbose=False
        )
        
        # Plot
        if filename is not None:
            filename = os.path.join(self.res_folder, "plots", filename)
        
        plotbib.plot_life_cycle(
            result,
            filename=filename,
            dpi=self.dpi,
            **kwargs
        )
        
        return result

    def plot_reference_spectrogram(self, save_path="spectrogram", **kwargs):
        """Plot a reference spectrogram over publication years.

        If the spectrogram has not yet been computed, it is created by calling
        :meth:`compute_reference_spectrogram` and then visualised with
        :func:`plotbib.plot_reference_spectrogram`.

        Parameters
        ----------
        save_path : str, default "spectrogram"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_reference_spectrogram`.
        """
        if not hasattr(self, "spectrogram_df"):
            self.compute_reference_spectrogram()
        save_path = os.path.join(self.res_folder, "plots", save_path)
        plotbib.plot_reference_spectrogram(self.spectrogram_df, save_path=save_path, **kwargs)

    def plot_ca_coutries_map(self, x="Number of documents", filename_prefix="country pefromance map", **kwargs):
        """Plot a choropleth map for countries with correspondence-analysis metrics.

        The method expects ``ca_country_counts_df`` to be available and will
        compute it via :meth:`count_ca_countries` if necessary. The data are
        then visualised with :func:`plotbib.save_plotly_choropleth_map`.

        Parameters
        ----------
        x : str, default "Number of documents"
            Column in ``ca_country_counts_df`` to use for colouring.
        filename_prefix : str, default "country pefromance map"
            Base filename (without extension) for saving the figure into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.save_plotly_choropleth_map`.
        """
        if not hasattr(self, "ca_country_counts_df"):
            self.count_ca_countries()
        if filename_prefix is not None:
            filename_prefix = os.path.join(self.res_folder, "plots", filename_prefix)

        plotbib.save_plotly_choropleth_map(self.ca_country_counts_df, x, filename_prefix=filename_prefix, 
                                           colormap=self.cmap, **kwargs)

    def plot_all_countries_map(self, x="Number of documents", filename_prefix="country collaboration map", **kwargs):
        
        """Plot a choropleth map of all collaborating countries.

        This uses the aggregated country statistics in
        ``all_countries_counts_df`` and the corresponding link table
        ``countries_links_df``. If the data are not present they must be
        computed beforehand.

        Parameters
        ----------
        x : str, default "Number of documents"
            Column in ``all_countries_counts_df`` to use for colouring.
        filename_prefix : str, default "country collaboration map"
            Base filename (without extension) for saving the figure into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.save_plotly_choropleth_map`.
        """
        plotbib.save_plotly_choropleth_map(self.all_countries_counts_df, x, filename_prefix=filename_prefix, 
                                           colormap=self.cmap, links_df=self.countries_links_df, **kwargs)
        
        

    # Bibliographic laws
    
    def lotka_law(self, author_col="Authors", filename_base="lotka law", **kwargs):
        """Compute and plot Lotka's law for author productivity.
    
        The method derives the Lotka frequency distribution from the current
        data frame, evaluates the fit and stores the results in
        ``lotka_df`` and ``lotka_stats_df`` before plotting the empirical
        distribution and the theoretical curve.
    
        If ``filename_base`` is not None, the plot is saved to
        ``<res_folder>/plots/<filename_base>.*`` (extensions determined by
        :mod:`plotbib`), and the tables ``lotka_df`` and ``lotka_stats_df``
        are written to an Excel file in
        ``<res_folder>/tables/<filename_base>.xlsx`` as two separate sheets.
    
        Parameters
        ----------
        author_col : str, default "Authors"
            Column containing author names.
        separator : str, default "; "
            Separator used between multiple authors in the same record.
        filename_base : str or None, default "lotka law"
            Base filename (without extension) for saving outputs. If None,
            results are not written to disk.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_lotka_distribution`.
        """
        # Compute Lotka distribution and fit statistics
        self.lotka_df = utilsbib.compute_lotka_distribution(
            self.df,
            author_col=author_col,
            separator=self.default_separator,
        )
        self.lotka_stats_df = utilsbib.evaluate_lotka_fit(self.lotka_df)
    
        plot_filename_base = None
        if filename_base is not None:
            # Plot base path (no extension, plotbib will add extensions)
            plot_filename_base = os.path.join(self.res_folder, "plots", filename_base)
    
            # Excel path for tables (two sheets)
            excel_path = os.path.join(self.res_folder, "tables", f"{filename_base}.xlsx")
            # Ensure parent folder exists (in case it was not created elsewhere)
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    
            with pd.ExcelWriter(excel_path) as writer:
                self.lotka_df.to_excel(
                    writer,
                    sheet_name="Lotka distribution",
                    index=False,
                )
                self.lotka_stats_df.to_excel(
                    writer,
                    sheet_name="Lotka stats",
                    index=False,
                )
                print(f"Saved to {excel_path}")
    
        # Plot (and optionally save) Lotka distribution
        plotbib.plot_lotka_distribution(
            self.lotka_df,
            filename_base=plot_filename_base,
            dpi=self.dpi,
            **kwargs,
        )

   
    def bradford_law(
        self,
        source_col="Source title",
        zone_count=3,
        lowercase=False,
        filename_base="bradford law",
        **kwargs,
    ):
        """Compute and plot Bradford's law for source dispersion.
    
        The method constructs the Bradford zones for the chosen source column,
        evaluates the theoretical fit and stores the results in ``bradford_df``
        and ``bradford_stats_df``. Two plots are produced: the empirical
        distribution and the zone representation.
    
        If ``filename_base`` is not None, the plots are saved to
        ``<res_folder>/plots/<filename_base> plot.*`` and
        ``<res_folder>/plots/<filename_base> zones.*`` (extensions determined by
        :mod:`plotbib`), and the tables ``bradford_df`` and ``bradford_stats_df``
        are written to an Excel file in
        ``<res_folder>/tables/<filename_base>.xlsx`` as two separate sheets.
    
        Parameters
        ----------
        source_col : str, default "Source title"
            Column containing the source (journal) titles.
        zone_count : int, default 3
            Number of Bradford zones to compute.
        lowercase : bool, default False
            If True, the source titles are converted to lower case before
            aggregation.
        filename_base : str or None, default "bradford law"
            Base filename (without extension) for saving outputs. If None,
            results are not written to disk.
        **kwargs :
            Additional keyword arguments forwarded to the underlying plotting
            routines in :mod:`plotbib`.
        """
        # Compute Bradford distribution and fit statistics
        self.bradford_df = utilsbib.compute_bradford_distribution(
            self.df,
            source_col=source_col,
            zone_count=zone_count,
            lowercase=lowercase,
        )
        self.bradford_stats_df = utilsbib.evaluate_bradford_fit(
            self.bradford_df,
            zone_count=zone_count,
        )
    
        F1_KEYS = {"color", "show_grid"}
        F2_KEYS = {
            "colors",
            "annotate_core",
            "show_labels",
            "label_rotation",
            "alt_label_col",
            "max_label_length",
            "show_grid",
        }
        kw1 = {k: kwargs[k] for k in F1_KEYS if k in kwargs}
        kw2 = {k: kwargs[k] for k in F2_KEYS if k in kwargs}
    
        plot_base_1 = None
        plot_base_2 = None
    
        if filename_base is not None:
            # Plot base paths (no extension; plotbib adds them)
            plot_base_1 = os.path.join(
                self.res_folder,
                "plots",
                f"{filename_base} plot",
            )
            plot_base_2 = os.path.join(
                self.res_folder,
                "plots",
                f"{filename_base} zones",
            )
    
            # Excel path for tables (two sheets)
            excel_path = os.path.join(
                self.res_folder,
                "tables",
                f"{filename_base}.xlsx",
            )
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    
            with pd.ExcelWriter(excel_path) as writer:
                self.bradford_df.to_excel(
                    writer,
                    sheet_name="Bradford distribution",
                    index=False,
                )
                self.bradford_stats_df.to_excel(
                    writer,
                    sheet_name="Bradford stats",
                    index=True,
                )
                print(f"Saved to {excel_path}")
    
        # Plot (and optionally save) Bradford distribution and zones
        plotbib.plot_bradford_distribution(
            self.bradford_df,
            title="Bradford's Law - Source Scattering",
            filename_base=plot_base_1,
            dpi=self.dpi,
            **kw1,
        )
        plotbib.plot_bradford_zones(
            self.bradford_df,
            title="Bradford's Law - Zones",
            filename_base=plot_base_2,
            dpi=self.dpi,
            **kw2,
        )

    def zipf_law(
        self,
        df_counts=None,
        items="words from abstract",
        filename_base="zipf law",
        **kwargs,
    ):
        """
        Compute and plot Zipf's law for item or term frequencies.
    
        Either an externally provided count table is used or the counts are
        looked up in the current ``mapping`` structure and, if necessary,
        computed first. The Zipf distribution and its fit statistics are stored
        in ``zipf_df`` and ``zipf_stats``.
    
        If ``filename_base`` is not None, the plot is saved to
        ``<res_folder>/plots/<filename_base>.*`` (extensions determined by
        :mod:`plotbib`), and the tables ``zipf_df`` and ``zipf_stats`` are
        written to an Excel file in
        ``<res_folder>/tables/<filename_base>.xlsx`` as two separate sheets.
    
        Parameters
        ----------
        df_counts : pandas.DataFrame or None, optional
            Pre-computed count table. If None, counts are taken from the
            internal mapping specified by ``items``.
        items : str, default "words from abstract"
            Key identifying which mapping entry to use when counts are not
            provided explicitly.
        filename_base : str or None, default "zipf law"
            Base filename (without extension) for saving outputs. If None,
            results are not written to disk.
        **kwargs :
            Additional keyword arguments forwarded to the internal counting
            (getter / counter) and plotting routines.
        """
    
        def _is_missing(value):
            """Return True if a mapping value is effectively missing."""
            if value is None:
                return True
            if isinstance(value, str) and value.strip() == "":
                return True
            try:
                return bool(pd.isna(value))
            except Exception:
                return False
    
        # Resolve counts if not provided
        if df_counts is None:
            try:
                d = self.mapping[items]
            except KeyError as exc:
                raise KeyError(
                    f'Unknown items key "{items}". '
                    f"Available keys are: {sorted(self.mapping.keys())}"
                ) from exc
    
            counts_df_attr = d.get("counts df")  # e.g. "words_abs_counts_df"
            counter_name = d.get("counter")      # e.g. "count_ngrams_abstract"
            getter_name = d.get("getter")        # e.g. "get_ngrams_abstract_stats"
    
            # 1) Use existing counts DataFrame if present
            if not _is_missing(counts_df_attr) and hasattr(self, counts_df_attr):
                df_counts = getattr(self, counts_df_attr)
    
            # 2) Otherwise, try getter (may compute counts & stats)
            if df_counts is None and not _is_missing(getter_name):
                getter = getattr(self, getter_name)
                res = getter(**kwargs)
                if isinstance(res, pd.DataFrame):
                    df_counts = res
                elif not _is_missing(counts_df_attr) and hasattr(self, counts_df_attr):
                    df_counts = getattr(self, counts_df_attr)
    
            # 3) Fallback: call counter directly if needed
            if df_counts is None and not _is_missing(counter_name):
                counter_fun = getattr(self, counter_name)
                res = counter_fun(**kwargs)
                if isinstance(res, pd.DataFrame):
                    df_counts = res
                elif not _is_missing(counts_df_attr) and hasattr(self, counts_df_attr):
                    df_counts = getattr(self, counts_df_attr)
    
            if df_counts is None:
                raise RuntimeError(
                    f"Could not obtain counts for items '{items}'. "
                    f"Tried getter '{getter_name}', counter '{counter_name}', "
                    f"and counts attribute '{counts_df_attr}'."
                )
    
        # Compute Zipf distribution and fit statistics
        self.zipf_df = utilsbib.compute_zipf_distribution_from_counts(df_counts)
        self.zipf_stats = utilsbib.evaluate_zipf_fit(self.zipf_df)
    
        # Prepare plot base (for plotbib) and Excel export path
        plot_filename_base = None
        if filename_base is not None:
            plot_filename_base = os.path.join(self.res_folder, "plots", filename_base)
    
            excel_path = os.path.join(
                self.res_folder,
                "tables",
                f"{filename_base}.xlsx",
            )
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    
            # Ensure stats are a DataFrame before exporting
            stats_obj = self.zipf_stats
            if isinstance(stats_obj, pd.DataFrame):
                stats_df = stats_obj
            elif isinstance(stats_obj, pd.Series):
                stats_df = stats_obj.to_frame().T
            elif isinstance(stats_obj, dict):
                # dict of scalars -> one-row DataFrame
                stats_df = pd.DataFrame([stats_obj])
            else:
                # generic fallback
                stats_df = pd.DataFrame({"value": [stats_obj]})
    
            with pd.ExcelWriter(excel_path) as writer:
                self.zipf_df.to_excel(
                    writer,
                    sheet_name="Zipf distribution",
                    index=False,
                )
                stats_df.to_excel(
                    writer,
                    sheet_name="Zipf stats",
                    index=False,
                )
    
        # Plot (and optionally save) Zipf distribution
        plotbib.plot_zipf_distribution(
            self.zipf_df,
            filename_base=plot_filename_base,
            dpi=self.dpi,
            **kwargs,
        )

    # Performance plots

    def plot_top_items(
        self,
        items,
        x="Number of documents",
        y="Total citations",
        kind="barh",
        top_n=None,
        default_properties=True,
        **kwargs,
    ):
        """
        Plot top items using preset templates, with optional top-N filtering.
    
        Notes
        -----
        - For kind="scatter" the usual defaults apply (size_col="H-index",
          color_col="Average year") unless the caller overrides them.
        - For kind in {"barh", "lollipop"} this function now:
            * sets color_by="Average year" by default (if not provided), and
            * aliases legacy kwargs:
                * color_col -> color_by
                * colormap  -> cmap
    
        Parameters
        ----------
        items : str
            Key in `self.mapping` describing which stats to plot.
        x, y, kind, top_n, default_properties, **kwargs
            See original docstring.
        """
        import os
    
        if items not in self.mapping:
            raise ValueError(f"Unknown item type: {items!r}")
    
        cfg = self.mapping[items]
        stats_attr = cfg["stats_attr"]
        get_stats = cfg["getter"]
        label = cfg["label"]
        default_label = cfg.get("default_label", label)
    
        # Ensure stats exist
        if not hasattr(self, stats_attr):
            getattr(self, get_stats)()
        df = getattr(self, stats_attr)
    
        # Default label for all kinds
        if default_properties and "label_col" not in kwargs:
            kwargs["label_col"] = default_label
    
        # Optional top-N selection (order_by logic preserved)
        order_by = kwargs.pop("order_by", cfg.get("order_by", x))
        if top_n and top_n > 0:
            if order_by in df.columns:
                df = (
                    df.dropna(subset=[order_by])
                    .sort_values(order_by, ascending=False)
                    .head(top_n)
                )
            else:
                df = df.head(top_n)
    
        base_fn = f"top_{items}_plot"
        filename = f"{base_fn}_{kind}"
        plot_path = os.path.join(self.res_folder, "plots", filename)
    
        # ----------------------------- Scatter ---------------------------------
        if kind == "scatter":
            # Keep existing defaults; alias color_by -> color_col if user provided it
            if "color_by" in kwargs and "color_col" not in kwargs:
                kwargs["color_col"] = kwargs.pop("color_by")
            if default_properties and "size_col" not in kwargs:
                kwargs["size_col"] = "H-index"
            if default_properties and "color_col" not in kwargs:
                kwargs["color_col"] = "Average year"
    
            kw = dict(kwargs)
            kw.pop("cmap", None)
            plotbib.plot_scatter(
                df,
                x,
                y,
                filename=plot_path,
                dpi=self.dpi,
                cmap=self.cmap,
                **kw,
            )
            return
    
        # --------------------- Non-scatter (barh / lollipop) --------------------
        # Default color_by if not provided (user can override, including None)
        if default_properties and "color_by" not in kwargs and "color_col" not in kwargs:
            kwargs["color_by"] = "Average year"
    
        # Legacy aliases
        if "color_col" in kwargs and "color_by" not in kwargs:
            kwargs["color_by"] = kwargs.pop("color_col")
        if "colormap" in kwargs and "cmap" not in kwargs:
            kwargs["cmap"] = kwargs.pop("colormap")
    
        fn = {
            "barh": plotbib.plot_barh,
            "lollipop": plotbib.plot_lollipop,
        }.get(kind)
        if fn is None:
            raise ValueError("kind must be one of \"barh\", \"lollipop\", or \"scatter\"")
    
        fn(
            df,
            x,
            label,
            filename=plot_path,
            dpi=self.dpi,
            cmap=self.cmap,
            default_color=self.default_color,
            **kwargs,
        )

    
    
    def plot_top_items_multi(
        self,
        items,
        x="Number of documents",
        y="Total citations",
        kind="barh",
        top_n=None,
        default_properties=True,
        **kwargs,
    ):
        """
        Loop over multiple item-types and plot each one in turn.
    
        Mirrors `plot_top_items` defaults and passes through `top_n`, `default_properties`,
        and any extra kwargs (including `order_by`).
    
        Parameters
        ----------
        items : Iterable[str]
            Stats keys defined in the same `mapping` as `plot_top_items`.
        x : str, default "Number of documents"
            X-axis column for each plot.
        y : str, default "Total citations"
            Y-axis column (used by "scatter").
        kind : {"barh", "lollipop", "scatter"}, default "barh"
            Plot style for each item.
        top_n : int or None, default None
            If set, apply top-N selection per item.
        default_properties : bool, default True
            Passed through to `plot_top_items`.
        **kwargs
            Forwarded to `plot_top_items` (e.g., order_by, label_col, size_col, color_col).
        """
        for item in items:
            try:
                self.plot_top_items(
                    item,
                    x=x,
                    y=y,
                    kind=kind,
                    top_n=top_n,
                    default_properties=default_properties,
                    **kwargs,
                )
            except Exception as e:
                logging.error(f"Failed to plot top items for {item!r}: {e}", exc_info=True)
    
    
    def scatter_plot_top_sources(self, x="Number of documents", y="Total citations", top_n=None, **kwargs):
        """
        Convenience wrapper for a scatter plot of top sources.
    
        Inherits mapping-driven defaults for labels and bubble encodings unless overridden via kwargs.
        Supports `top_n` for limiting the number of points.
        """
        self.plot_top_items("sources", x=x, y=y, kind="scatter", top_n=top_n, default_properties=True, **kwargs)
    
    
    def scatter_plot_top_authors(self, x="Number of documents", y="Total citations", top_n=None, **kwargs):
        """
        Convenience wrapper for a scatter plot of top authors.
    
        Supports `top_n` for limiting the number of points.
        """
        self.plot_top_items("authors", x=x, y=y, kind="scatter", top_n=top_n, default_properties=True, **kwargs)
    
    
    def scatter_plot_top_countries(self, x="Number of documents", y="Total citations", top_n=None, **kwargs):
        """
        Convenience wrapper for a scatter plot of top countries.
    
        Supports `top_n` for limiting the number of points.
        """
        self.plot_top_items("all countries", x=x, y=y, kind="scatter", top_n=top_n, default_properties=True, **kwargs)



    
    
    
    def visualize_text(self, items, kind="cloud", x="Number of documents", filename="wordcloud", top_n=20, **kwargs):
        
        """Visualise the most frequent textual items as a word cloud or treemap.

        This is a convenience wrapper around :func:`plotbib.visualize_text`
        that pulls the appropriate count table from the internal ``mapping``
        structure and applies basic filtering.

        Parameters
        ----------
        items : str
            Name of the mapping entry to visualise (for example "authors" or
            "author keywords").
        kind : {"cloud", "treemap"}, default "cloud"
            Type of visualisation to produce.
        x : str, default "Number of documents"
            Column used to rank items before visualisation.
        filename : str, default "wordcloud"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        top_n : int, default 20
            Number of top items to include.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.visualize_text`.
        """
        if items not in self.mapping:
            raise ValueError(f"Unknown item type: {items!r}")
    
        fn = {"cloud": plotbib.plot_wordcloud, "treemap": plotbib.plot_treemap}[kind]
    
        cfg = self.mapping[items]
        stats_attr = cfg["stats_attr"]
        get_stats = cfg["getter"]
    
        if not hasattr(self, stats_attr):
            getattr(self, get_stats)(top_n=top_n)
        df = getattr(self, stats_attr).head(top_n)
        
        filename = os.path.join(self.res_folder, "plots", filename + "_" + kind + "_" + items)
        if kind == "cloud":
            fn(df, filename=filename, dpi=self.dpi, colormap=self.cmap, **kwargs)
        else:
            fn(df, filename=filename, dpi=self.dpi, cmap=self.cmap, **kwargs)
                
    def visualize_text_multi(self, items, kind="cloud", x="Number of documents", filename="wordcloud", top_n=20, **kwargs):
        """Visualise text statistics for multiple item types.

        The method simply loops over ``items`` and calls
        :meth:`visualize_text` for each, using a file-name suffix to keep
        the outputs separate.

        Parameters
        ----------
        items : sequence of str
            Sequence of mapping keys to visualise.
        kind : {"cloud", "treemap"}, default "cloud"
            Type of visualisation to produce.
        x : str, default "Number of documents"
            Column used to rank items before visualisation.
        filename : str, default "wordcloud"
            Base filename (without extension) for saving the plots into
            ``<res_folder>/plots``.
        top_n : int, default 20
            Number of top items to include per visualisation.
        **kwargs :
            Additional keyword arguments forwarded to :meth:`visualize_text`.
        """
        for item in items:
            try:
                self.visualize_text(item, kind=kind, x=x, filename=filename, top_n=top_n, **kwargs)
            except Exception as e:
                logging.error(f"Failed to plot top items for {item!r}: {e}", exc_info=True)
                
    def plot_thematic_map(self, G=None, items="author keywords", recompute=False, partition_attr="walktrap", max_dot_size=200, 
                          quadrant_labels=False, items_per_cluster=3,
                          cmap_name="viridis",  figsize=(8, 6),  max_clusters=None,
                          min_cluster_size=5, include_cluster_label=False,
                          color_df=None, color_col=None, save_plot_base="thematic map",
                          ax=None, item_sep="\n", **kwargs):
        
    
        """Plot a thematic map based on a co-occurrence network.

        The network is obtained from the global ``co_mapping`` object (for
        example the keyword co-occurrence network) and, if necessary,
        recomputed. The map itself is produced with
        :func:`plotbib.plot_thematic_map`.

        Parameters
        ----------
        G : networkx.Graph or None, optional
            Pre-built network. If None, the appropriate network is taken from
            ``co_mapping``.
        items : str, default "author keywords"
            Name of the network to use from ``co_mapping``.
        recompute : bool, default False
            If True, rebuild the network even if it already exists.
        partition_attr : str, default "walktrap"
            Attribute name containing the community partition to highlight.
        max_dot_size : int, default 200
            Maximum node size in the scatter plot.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_thematic_map`.
        """
        if G is None:
            d = co_mapping[items]
            if not hasattr(self, d["net_attr"]) or recompute:
                getattr(self, d["getter"])(**kwargs)
            G = getattr(self, d["net_attr"])

        if save_plot_base is not None:
            save_plot_base = os.path.join(self.res_folder, "plots", partition_attr + "_" + save_plot_base)

        plotbib.plot_thematic_map(G, partition_attr, max_dot_size=max_dot_size, 
                              quadrant_labels=quadrant_labels, items_per_cluster=items_per_cluster,
                              cmap_name=self.cmap, figsize=figsize, max_clusters=max_clusters,
                              min_cluster_size=min_cluster_size, include_cluster_label=include_cluster_label,
                              color_df=color_df, color_col=color_col, save_plot_base=save_plot_base,
                              dpi=self.dpi, ax=ax, item_sep=item_sep)

    def plot_word_map(
        self,
        figsize: tuple = (10, 8),
        title: str = "Word Map",
        filename_base: str = "factorial word map",
        marker_size: int = 50,
        term_fontsize: int = 8,
        title_fontsize: int = 12,
        axis_label_fontsize: int = 10,
        tick_label_fontsize: int = 8,
        xlabel: str = "Dim 1",
        ylabel: str = "Dim 2",
        show_legend: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot a conceptual word map using correspondence analysis.
    
        This method expects that conceptual structure statistics have been
        computed beforehand (for example via ``conceptual_structure_analysis``).
        It then forwards the relevant data to :func:`plotbib.plot_word_map`.
    
        Term labels are normalised with ``utilsbib._balance_closing_parenthesis``,
        so unmatched "(" are balanced with a closing ")" (same behaviour as in
        ``utilsbib.count_occurrences``).
    
        Parameters
        ----------
        figsize : tuple of float, default (10, 8)
            Figure size passed to the plotting function.
        title : str, default "Word Map"
            Title of the plot.
        filename_base : str, default "factorial word map"
            Base filename (without extension) used when saving the figure.
        marker_size : int, default 50
            Size of scatter markers.
        term_fontsize : int, default 8
            Font size for term annotations.
        title_fontsize : int, default 12
            Font size for the title.
        axis_label_fontsize : int, default 10
            Font size for axis labels.
        tick_label_fontsize : int, default 8
            Font size for tick labels.
        xlabel : str, default "Dim 1"
            Label for the x-axis.
        ylabel : str, default "Dim 2"
            Label for the y-axis.
        show_legend : bool, default True
            Whether to show the cluster legend.
        **kwargs :
            Additional keyword arguments forwarded to
            :meth:`conceptual_structure_analysis` when it is called.
        """
        if not hasattr(self, "conceptual_structure_d"):
            self.conceptual_structure_analysis(**kwargs)
    
        # Balance parentheses in terms (same behaviour as in count_occurrences)
        raw_terms = self.conceptual_structure_d["terms"]
        terms = [utilsbib._balance_closing_parenthesis(str(t)) for t in raw_terms]
    
        filename_base = os.path.join(self.res_folder, "plots", filename_base)
    
        plotbib.plot_word_map(
            embeddings=self.conceptual_structure_d["term_embeddings"],
            terms=terms,
            labels=self.conceptual_structure_d["term_labels"],
            figsize=figsize,
            title=title,
            filename_base=filename_base,
            dpi=self.dpi,
            cmap=self.cmap,
            marker_size=marker_size,
            term_fontsize=term_fontsize,
            title_fontsize=title_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            xlabel=xlabel,
            ylabel=ylabel,
            show_legend=show_legend,
        )

    
    def plot_topic_dendrogram(
        self,
        method: str = "ward",
        metric: str = "euclidean",
        figsize: tuple = (10, 8),
        title: str = "Topic Dendrogram",
        filename_base: str = "topic dendrogram",
        xlabel: str = "Terms",
        ylabel: str = "Distance",
        title_fontsize: int = 12,
        axis_label_fontsize: int = 10,
        tick_label_fontsize: int = 8,
        leaf_label_fontsize: int = 8,
        **kwargs,
    ):
        """
        Plot a dendrogram of terms based on conceptual-structure embeddings.
    
        This method uses ``self.conceptual_structure_d``, which is created by
        :meth:`conceptual_structure_analysis`. If that attribute is not yet
        available, the method first calls ``self.conceptual_structure_analysis(**kwargs)``.
        Term labels are post-processed with
        ``utilsbib._balance_closing_parenthesis`` so that unmatched "(" are
        balanced (same behaviour as in ``count_occurrences``).
    
        Parameters
        ----------
        method : str, default "ward"
            Linkage method used for hierarchical clustering.
        metric : str, default "euclidean"
            Distance metric between term vectors (forwarded to the plotting
            helper in :mod:`plotbib` if supported).
        figsize : tuple of float, default (10, 8)
            Figure size passed to the plotting function.
        title : str, default "Topic Dendrogram"
            Plot title.
        filename_base : str, default "topic dendrogram"
            Base name for saving the figure. The full path is constructed as
            ``self.res_folder / "plots" / filename_base``.
        xlabel : str, default "Terms"
            Label for the x-axis.
        ylabel : str, default "Distance"
            Label for the y-axis.
        title_fontsize : int, default 12
            Font size for the plot title.
        axis_label_fontsize : int, default 10
            Font size for axis labels.
        tick_label_fontsize : int, default 8
            Font size for tick labels.
        leaf_label_fontsize : int, default 8
            Font size for dendrogram leaf labels.
        **kwargs :
            Additional keyword arguments forwarded to
            :meth:`conceptual_structure_analysis` when it is called.
        """
        # Ensure conceptual structure has been computed
        if not hasattr(self, "conceptual_structure_d"):
            self.conceptual_structure_analysis(**kwargs)
    
        # Balance parentheses in terms (same behaviour as in count_occurrences)
        raw_terms = self.conceptual_structure_d["terms"]
        terms = [utilsbib._balance_closing_parenthesis(t) for t in raw_terms]
    
        # Build full filename base
        filename_base = os.path.join(self.res_folder, "plots", filename_base)
    
        # Delegate plotting to plotbib helper
        plotbib.plot_topic_dendrogram(
            embeddings=self.conceptual_structure_d["term_embeddings"],
            terms=terms,
            method=method,
            metric=metric,
            figsize=figsize,
            title=title,
            filename_base=filename_base,
            dpi=self.dpi,
            xlabel=xlabel,
            ylabel=ylabel,
            title_fontsize=title_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            leaf_label_fontsize=leaf_label_fontsize,
        )

        
    def plot_trend_topics(
        self,
        df: pd.DataFrame = None,
        items: str = "Author Keywords",
        min_docs: int = 3,
        regex_filter: str | None = None,
        top_n_year: int = 3,
        color_by: str = "Total citations",
        item_col: str = "Item",
        figsize: tuple = (10, 6),
        filename: str = "trend topics",
        title: str | None = None,
        title_fontsize: int = 12,
        label_fontsize: int = 10,
        tick_fontsize: int = 8,
        median_rounding: str | None = None,
        override: bool = True,
        log_size: bool = True,
        **kwargs,
    ):
        """
        Plot trend topics ordered by median year, with Q1–Q3 spans and a size legend.
    
        Changes
        -------
        - Thinner Q1–Q3 lines: `line_width=1.0`.
        - Grey span lines (requires tiny patch in `plot_item_time_stats`, see below).
    
        Returns
        -------
        matplotlib.figure.Figure
        """
        import os
        import re
        import numpy as np
        import pandas as pd
    
        d = self.mapping[items.lower()]
        if (df is None) or override:
            if not hasattr(self, d["stats_attr"]):
                getattr(self, d["getter"])(**kwargs)
            df = getattr(self, d["stats_attr"])
            item_col = d["label"]
    
        data = df.copy()
        cols = {c.lower(): c for c in data.columns}
    
        def _pick(*names: str) -> str | None:
            for n in names:
                if n in data.columns:
                    return n
                if n.lower() in cols:
                    return cols[n.lower()]
            return None
    
        size_col = _pick("Number of documents", "n_docs") or "Number of documents"
        if size_col not in data.columns:
            raise KeyError('Size column not found (expected "Number of documents" or "n_docs").')
    
        year_point_col = _pick("Year", "Publication Year", "publication_year")
    
        med_col = _pick("Median Year", "Median", "median_year", "median")
        if med_col is None:
            if year_point_col is None:
                raise KeyError('Median not found and no "Year" to compute it.')
            med_per_item = data.groupby(item_col)[year_point_col].median()
            data = data.merge(med_per_item.rename("Median Year"), on=item_col, how="left")
        elif med_col != "Median Year":
            data["Median Year"] = data[med_col]
    
        q1_col = _pick("Q1", "Q1 Year", "q1_year", "q1")
        q3_col = _pick("Q3", "Q3 Year", "q3_year", "q3")
        if q1_col is None or q3_col is None:
            if year_point_col is not None:
                q1_series = data.groupby(item_col)[year_point_col].quantile(0.25).rename("Q1")
                q3_series = data.groupby(item_col)[year_point_col].quantile(0.75).rename("Q3")
                data = data.merge(q1_series, on=item_col, how="left").merge(q3_series, on=item_col, how="left")
        else:
            if q1_col != "Q1":
                data["Q1"] = data[q1_col]
            if q3_col != "Q3":
                data["Q3"] = data[q3_col]
    
        for c in ("Median Year", "Q1", "Q3"):
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")
    
        per_item = (
            data.groupby(item_col)
            .agg(total_docs=(size_col, "sum"), median_year_val=("Median Year", "median"))
            .dropna(subset=["median_year_val"])
        )
        if min_docs and min_docs > 0:
            per_item = per_item.loc[per_item["total_docs"] >= min_docs]
    
        if regex_filter:
            pat = re.compile(regex_filter, flags=re.I)
            per_item = per_item[per_item.index.to_series().astype(str).apply(lambda s: bool(pat.search(s)))]
    
        if top_n_year and top_n_year > 0:
            tmp = per_item.assign(_my=per_item["median_year_val"].astype(float).round().astype("Int64"))
            per_item = (
                tmp.sort_values(["_my", "total_docs"], ascending=[False, False])
                   .groupby("_my", group_keys=False)
                   .head(top_n_year)
                   .drop(columns=["_my"])
            )
        if per_item.empty:
            raise ValueError("No items left to plot after filtering.")
    
        order = per_item.sort_values(["median_year_val", "total_docs"], ascending=[False, False]).index.tolist()
        data = data[data[item_col].isin(order)].copy()
    
        filename = os.path.join(self.res_folder, "plots", f"{filename}_{items}")
    
        fig = plotbib.plot_item_time_stats(
            data,
            year_col="Median Year",
            item_order=order,
            color_by=color_by,
            item_col=item_col,
            figsize=figsize,
            dpi=self.dpi,
            filename=filename,
            title=title,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            median_rounding=median_rounding,
            color_scheme=self.cmap,
            log_size=log_size,
            # thinner, subtler span lines
            line_width=1.0,
            line_alpha=0.9,
            # legend
            size_legend=True,
            size_legend_title="Number of documents",
            size_legend_facecolor="black",
            size_legend_edgecolor="black",
            size_legend_bottom=0.16,
        )
        return fig
  
   
    def plot_items_production_over_time(
        self,
        items: str,
        file_name: Optional[str] = None,
        top_n: Optional[int] = None,
        compute_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        y_label: Optional[str] = None,
    ):
        """
        Compute and plot production-over-time for a mapping key, with robust kwarg handling.
    
        - Stores the computed table in self.<items>_production_over_time_df.
        - Saves the table to "<res_folder>/tables/<file_name>.xlsx" (if self.res_folder).
        - Moves "min_docs" / "min_docs_per_item" from plot_kwargs to compute_kwargs.
        - Filters unknown plot kwargs to avoid "unexpected keyword" errors.
    
        Parameters
        ----------
        items : str
            Key into self.mapping (e.g., "authors", "keywords", "countries").
        file_name : str, optional
            Base filename (no extension). Defaults to mapping savepath or "<items>_production_over_time".
        top_n : int, optional
            Forwarded to utilsbib.compute_item_time_stats.
        compute_kwargs : dict, optional
            Extra kwargs for utilsbib.compute_item_time_stats.
        plot_kwargs : dict, optional
            Extra kwargs for plotbib.plot_item_time_stats. Unknown keys are ignored.
        y_label : str, optional
            Y-axis label; defaults to self.mapping[items]["label"].
    
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        import os
        import re
    
        # Resolve mapping-driven inputs
        v = self.mapping[items]["time production var"]
        vp = "Processed " + v
        if vp in self.df.columns:
            v = vp
        default_fname = self.mapping[items].get("time production savepath", f"{items}_production_over_time")
        file_name = file_name or default_fname
        if y_label is None:
            y_label = self.mapping[items]["label"]
    
        # Prepare kwargs
        compute_kwargs = dict(compute_kwargs or {})
        plot_kwargs = dict(plot_kwargs or {})
    
        # Expose top_n
        if top_n is not None:
            compute_kwargs["top_n"] = int(top_n)
    
        # Redirect min_docs from plot -> compute to match compute_item_time_stats API
        # Accept both "min_docs" and "min_docs_per_item" spellings
        if "min_docs" in plot_kwargs:
            compute_kwargs["min_docs_per_item"] = int(plot_kwargs.pop("min_docs"))
        if "min_docs_per_item" in plot_kwargs:
            compute_kwargs["min_docs_per_item"] = int(plot_kwargs.pop("min_docs_per_item"))
        if "min_docs" in compute_kwargs:
            compute_kwargs["min_docs_per_item"] = int(compute_kwargs.pop("min_docs"))
    
        # Compute table
        if self.db == "oa":
            list_separators = r"\|"
        else:
            list_separators = self.default_separator
        df_out = utilsbib.compute_item_time_stats(self.df, v, list_separators=list_separators, **compute_kwargs)
        df_out[df_out.columns[0]] = df_out[df_out.columns[0]].map(utilsbib._balance_closing_parenthesis)
    
        # Store under an item-specific attribute
        slug = re.sub(r"\W+", "_", str(items).strip().lower()).strip("_")
        attr_name = f"{slug}_production_over_time_df"
        setattr(self, attr_name, df_out)
    
        # Defaults: cmap/dpi/savepath
        plot_kwargs.setdefault("cmap", getattr(self, "cmap", "viridis"))
        plot_kwargs.setdefault("dpi", getattr(self, "dpi", 600))
    
        # Build output dirs and save table
        if getattr(self, "res_folder", None):
            plots_dir = os.path.join(self.res_folder, "plots")
            tables_dir = os.path.join(self.res_folder, "tables")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
    
            table_path = os.path.join(tables_dir, f"{file_name}.xlsx")
            try:
                df_out.to_excel(table_path, index=False)
            except Exception:
                # Fallback if Excel writer unavailable
                df_out.to_csv(os.path.join(tables_dir, f"{file_name}.csv"), index=False)
    
            plot_kwargs.setdefault("savepath", os.path.join(plots_dir, file_name))
    
        # Whitelist known plot kwargs to avoid TypeError
        allowed_plot_keys = {
            "item_order", "size_col", "color_col", "cmap", "size_min", "size_max",
            "line_alpha", "line_width", "y_label", "wrap_width", "savepath", "dpi",
            "title_fontsize", "label_fontsize", "tick_fontsize",
        }
        safe_plot_kwargs = {k: v for k, v in plot_kwargs.items() if k in allowed_plot_keys}
        
        # Set smaller default font sizes for better readability
        safe_plot_kwargs.setdefault("tick_fontsize", 8)
        safe_plot_kwargs.setdefault("label_fontsize", 10)
    
        # Plot
        return plotbib.plot_item_time_stats(df_out, y_label=y_label, **safe_plot_kwargs)


    # topic modelling
    
    def plot_topic_visualization(
        self,
        *,
        kind: str,
        df_out: Optional[pd.DataFrame] = None,
        topics_df: Optional[pd.DataFrame] = None,
        doc_col: Optional[str] = None,
        topic_col: Optional[str] = None,
        doc_weight_col: Optional[str] = None,
        time_col: Optional[str] = None,
        term_col: Optional[str] = None,
        term_weight_col: Optional[str] = None,
        topic_id: Optional[Any] = None,
        top_n_terms: int = 15,
        top_n_topics: int = 10,
        max_docs: int = 200,
        normalize: bool = True,
        figsize: Tuple[float, float] = (9, 6),
        cmap: Optional[str] = None,
        title: Optional[str] = None,
        grid: bool = False,
        filename_base: Optional[str] = None,
        dpi: Optional[int] = None,
        topic_label_offset: int = 0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Thin wrapper around utilsbib.plot_topic_visualization that honors class defaults.
    
        Behavior
        --------
        - Uses self.cmap (fallback "viridis") unless `cmap` is provided.
        - Uses self.dpi (fallback 600) unless `dpi` is provided.
        - If `filename_base` is None and `self.res_folder` is set, saves to:
          {self.res_folder}/plots/topics/{kind}[_topic-{topic_id}].{png,svg,pdf}
    
        Returns
        -------
        (fig, ax)
        """
    
        _cmap = cmap or getattr(self, "cmap", "viridis")
        _dpi = int(dpi or getattr(self, "dpi", 600))
    
        _filename = filename_base
        if _filename is None and getattr(self, "res_folder", None):
            base = kind + (f"_topic-{topic_id}" if topic_id is not None else "")
            out_dir = os.path.join(self.res_folder, "plots", "topics")
            os.makedirs(out_dir, exist_ok=True)
            _filename = os.path.join(out_dir, base)
    
        fig, ax = plotbib.plot_topic_visualization(
            kind=kind,
            df_out=self.topic_assignment_df,
            topics_df=self.topics_df,
            doc_col=doc_col,
            topic_col=topic_col,
            doc_weight_col=doc_weight_col,
            time_col=time_col,
            term_col=term_col,
            term_weight_col=term_weight_col,
            topic_id=topic_id,
            top_n_terms=top_n_terms,
            top_n_topics=top_n_topics,
            max_docs=max_docs,
            normalize=normalize,
            figsize=figsize,
            cmap=_cmap,
            title=title,
            grid=grid,
            filename_base=_filename,
            dpi=_dpi,
            topic_label_offset=topic_label_offset,
        )
        return fig, ax


    # general co-occurrence   
    def plot_coocurrence(
        self,
        items,                               # str key into mapping OR a NetworkX graph
        partition_attrs=["walktrap", "edge_betweenness", "label_propagation", "girvan_newman", "kernighan_lin"],
        overlay_color_attr=None,
        overlay_size_attr=None,
        filename_prefix="cooccurrence",
        cluster_labels=None,
        **kwargs,
    ):
        """
        "Plot co-occurrence networks from either a graph or a named 'items' key via self.mapping.
        
        If `items` is a string, the function looks up a graph in `self.mapping[items]` using
        either an attribute name under 'coocurence network' (graph stored on self) or a
        getter method name under 'get coocurrence' (method on self returning a graph).
        If `items` is already a NetworkX graph, it is used directly.
    
        Two kinds of plots are produced:
          1) Partition views (one per value in `partition_attrs`), colored by communities,
             titled 'Clusters by <partition_attr>' and optionally labeled via `cluster_labels`.
          2) One overlay view colored by `overlay_color_attr` (default: 'Average year') and
             sized by `overlay_size_attr` (default: 'Number of documents').
    
        Style & saving:
          - Uses self.dpi, self.cmap (continuous), self.cmap_disc (discrete).
          - If self.res_folder is set, saves to <self.res_folder>/plots using plotbib.save_plot()
            with reasonable filenames incorporating `items`, partition name, and overlay specs.
    
        Returns
        -------
        dict
            {'overlay': (fig, ax, pos), 'partition:<attr>': (fig, ax, pos), ...}
        """
        import os
        import re
        import matplotlib.pyplot as plt
        from biblium import plotbib
        try:
            import networkx as nx
        except Exception:
            nx = None  # only used for isinstance checks
    
        # ---- resolve the graph G from `items` ----
        G = None
        items_name = None
    
        if nx is not None and hasattr(items, "nodes") and hasattr(items, "edges"):
            # NetworkX-like object passed directly
            G = items
            items_name = "graph"
        elif isinstance(items, str):
            items_name = items
            if not isinstance(self.mapping, dict) or items not in self.mapping:
                raise KeyError(f"Could not resolve items='{items}' via self.mapping.")
            cfg = self.mapping[items]
    
            # Prefer a stored network attribute if available
            net_key_variants = ["coocurence network", "cooccurrence network"]
            get_key_variants = ["get coocurrence", "get cooccurrence"]
    
            net_attr = next((cfg[k] for k in net_key_variants if k in cfg), None)
            get_name = next((cfg[k] for k in get_key_variants if k in cfg), None)
    
            if net_attr and hasattr(self, net_attr):
                G = getattr(self, net_attr)
            elif get_name and hasattr(self, get_name):
                getter = getattr(self, get_name)
                G = getter() if callable(getter) else getter
            else:
                raise KeyError(
                    f"No graph source in mapping for '{items}'. "
                    f"Expected keys {net_key_variants + get_key_variants}."
                )
        else:
            raise TypeError("`items` must be a NetworkX graph or a string key present in self.mapping.")
    
        # ---- class-level style defaults ----
        dpi = getattr(self, "dpi", 300)
        cmap_cont = getattr(self, "cmap", "viridis")
        cmap_disc = getattr(self, "cmap_disc", "tab10")
    
        # ---- saving setup (self.res_folder controls saving) ----
        res_folder = getattr(self, "res_folder", None)
        save_flag = res_folder is not None
        if save_flag:
            plots_dir = os.path.join(res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
        else:
            plots_dir = None
    
        # ---- helpers ----
        def _safe_name(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "attr"
    
        def _labels_for(part_attr):
            """
            Return labels mapping for the given partition attribute.
            Accepts:
              - cluster_labels: {cluster_id: label} when only one partition is used
              - cluster_labels: {partition_attr: {cluster_id: label}} for multiple partitions
            """
            if cluster_labels is None:
                return None
            if isinstance(cluster_labels, dict) and any(isinstance(v, dict) for v in cluster_labels.values()):
                return cluster_labels.get(part_attr, None)
            return cluster_labels
    
        out = {}
    
        # --------- Partition views ----------
        if partition_attrs:
            if isinstance(partition_attrs, str):
                partition_attrs = [partition_attrs]
    
            for part_attr in partition_attrs:
                # Per-partition kwargs (respect user size choice if provided)
                part_kwargs = dict(kwargs)
                part_size_attr = part_kwargs.pop("size_attr", part_kwargs.pop("partition_size_attr", "Number of documents"))
    
                fig, ax, pos = plotbib.plot_network(
                    G,
                    partition_attr=part_attr,
                    color_attr=None,  # enforce partition coloring
                    cluster_labels=_labels_for(part_attr),
                    size_attr=part_size_attr,
                    dpi=dpi,
                    cmap_name_continuous=cmap_cont,
                    cmap_name_discrete=cmap_disc,
                    **part_kwargs,
                )
                out[f"partition:{part_attr}"] = (fig, ax, pos)
    
                # Save via plotbib.save_plot if requested
                if save_flag and plots_dir:
                    base = os.path.join(
                        plots_dir,
                        f"{_safe_name(filename_prefix)}_{_safe_name(items_name)}_partition_{_safe_name(part_attr)}",
                    )
                    plt.figure(fig.number)  # ensure current fig for save_plot
                    plotbib.save_plot(base, dpi=dpi)
    
        # --------- Overlay view ----------
        # Precedence: explicit args > kwargs > defaults
        overlay_kwargs = dict(kwargs)
        if overlay_color_attr is not None:
            overlay_color = overlay_color_attr
            overlay_kwargs.pop("color_attr", None)
        else:
            overlay_color = overlay_kwargs.pop("color_attr", "Average year")
    
        if overlay_size_attr is not None:
            overlay_size = overlay_size_attr
            overlay_kwargs.pop("size_attr", None)
        else:
            overlay_size = overlay_kwargs.pop("size_attr", "Number of documents")
    
        fig_o, ax_o, pos_o = plotbib.plot_network(
            G,
            partition_attr=None,
            color_attr=overlay_color,
            size_attr=overlay_size,
            dpi=dpi,
            cmap_name_continuous=cmap_cont,
            cmap_name_discrete=cmap_disc,
            **overlay_kwargs,
        )
        out["overlay"] = (fig_o, ax_o, pos_o)
    
        if save_flag and plots_dir:
            base = os.path.join(
                plots_dir,
                f"{_safe_name(filename_prefix)}_{_safe_name(items_name)}_overlay_{_safe_name(overlay_color)}_{_safe_name(overlay_size)}",
            )
            plt.figure(fig_o.number)
            plotbib.save_plot(base, dpi=dpi)
    
        return out

    # specific co-occurrences
    def plot_co_citation_network(
        self,
        partition_attrs=["walktrap", "edge_betweenness", "label_propagation", "girvan_newman", "kernighan_lin"],
        overlay_color_attr="Average year",
        overlay_size_attr="Number of documents",
        filename_prefix="co_ciation",
        cluster_labels=None,
        **kwargs,
    ):
        """
        "Plot a co-citation network by delegating to plot_coocurrence, resolving the graph from mapping['references'].
    
        Notes
        -----
        - The co-citation graph is deduced via the global `mapping` entry 'references'.
        - Filenames for saved figures start with 'co_ciation' by default.
        - Uses class styling and saving behavior implemented in `plot_coocurrence`."
        """
        return self.plot_coocurrence(
            items="references",
            partition_attrs=partition_attrs,
            overlay_color_attr=overlay_color_attr,
            overlay_size_attr=overlay_size_attr,
            filename_prefix=filename_prefix,
            cluster_labels=cluster_labels,
            **kwargs,
        )

    def plot_co_authorship_network(
        self,
        partition_attrs=["walktrap", "edge_betweenness", "label_propagation", "girvan_newman", "kernighan_lin"],
        overlay_color_attr="Average year",
        overlay_size_attr="Number of documents",
        filename_prefix="co_authorship",
        cluster_labels=None,
        **kwargs,
    ):
        """
        "Plot a co-authorship network by delegating to plot_coocurrence, resolving the graph from mapping['authors'].
    
        Notes
        -----
        - The co-authorship graph is deduced via the global mapping entry 'authors'.
        - Filenames for saved figures start with 'co_authorship' by default.
        - Uses class styling and saving behavior implemented in plot_coocurrence."
        """
        return self.plot_coocurrence(
            items="authors",
            partition_attrs=partition_attrs,
            overlay_color_attr=overlay_color_attr,
            overlay_size_attr=overlay_size_attr,
            filename_prefix=filename_prefix,
            cluster_labels=cluster_labels,
            **kwargs,
        )

    def plot_keyword_coocurrence_network(
        self,
        kind="author",                       # "author", "index", or "both"
        partition_attrs=["walktrap", "edge_betweenness", "label_propagation", "girvan_newman", "kernighan_lin"],
        overlay_color_attr="Average year",
        overlay_size_attr="Number of documents",
        filename_prefix=None,
        cluster_labels=None,
        **kwargs,
    ):
        """
        "Plot keyword co-occurrence network using a key resolved from the global `mapping`."
    
        Resolution logic (no special-case lists; we trust user's mapping names):
          - kind="author" → pick the (single) mapping key that contains "author" AND "keyword".
          - kind="index"  → pick the (single) mapping key that contains "index"  AND "keyword".
          - kind="both"   → pick ONE merged-key that contains BOTH "author" AND "index" AND "keyword".
                            (Runs once on the merged network; no double work.)
    
        Saving/Styling:
          - Delegates to `self.plot_coocurrence` which handles saving (via plotbib.save_plot) when `self.res_folder` is set,
            and uses class styling (`self.dpi`, `self.cmap`, `self.cmap_disc`).
        """
        if kind not in {"author", "index", "both"}:
            raise ValueError('kind must be one of {"author", "index", "both"}')
    
        keys = self.mapping.keys()
        lower = {k: k.lower() for k in keys}
    
        def _pick_one(predicate, err_msg):
            candidates = [k for k in keys if predicate(lower[k])]
            if not candidates:
                raise KeyError(f"{err_msg} Available keys: {keys}")
            return sorted(candidates)[0]  # deterministic choice without extra heuristics
    
        if kind == "author":
            items_key = _pick_one(
                lambda s: ("author" in s) and ("keyword" in s),
                'No mapping entry found that contains both "author" and "keyword".',
            )
            default_prefix = "keyword_coocurrence_author"
        elif kind == "index":
            items_key = _pick_one(
                lambda s: ("index" in s) and ("keyword" in s),
                'No mapping entry found that contains both "index" and "keyword".',
            )
            default_prefix = "keyword_coocurrence_index"
        else:  # kind == "both"
            items_key = _pick_one(
                lambda s: ("author" in s) and ("index" in s) and ("keyword" in s),
                'No merged keywords mapping entry found that contains "author", "index", and "keyword".',
            )
            default_prefix = "keyword_coocurrence"
    
        if filename_prefix is None:
            filename_prefix = default_prefix
    
        return self.plot_coocurrence(
            items=items_key,
            partition_attrs=partition_attrs,
            overlay_color_attr=overlay_color_attr,
            overlay_size_attr=overlay_size_attr,
            filename_prefix=filename_prefix,
            cluster_labels=cluster_labels,
            **kwargs,
        )


    def plot_ngrams_coocurrence_network(
        self,
        source="abstract",                   # "abstract", "title", or "combined text"
        partition_attrs=["walktrap", "edge_betweenness", "label_propagation", "girvan_newman", "kernighan_lin"],
        overlay_color_attr="Average year",
        overlay_size_attr="Number of documents",
        filename_prefix=None,
        cluster_labels=None,
        **kwargs,
    ):
        """
        "Plot an n-grams co-occurrence network by selecting a mapping key that contains
        'abstract', 'title', or 'combined text' (case-insensitive), with no preference for 'processed'.
        Assumes the target network is already available via the global `mapping`."
    
        Returns the dict from `self.plot_coocurrence`.
        """
        if source not in {"abstract", "title", "combined text"}:
            raise ValueError('source must be one of {"abstract", "title", "combined text"}')
    
        token = source.lower()
        keys = self.mapping.keys()
        candidates = [k for k in keys if token in k.lower()]
        if not candidates:
            raise KeyError(f"No mapping entry contains {source!r}. Available keys: {keys}")
    
        # No 'processed' preference — pick a stable alphabetical match
        items_key = sorted(candidates)[0]
    
        if filename_prefix is None:
            filename_prefix = f"ngrams_coocurrence_{token.replace(' ', '_')}"
    
        return self.plot_coocurrence(
            items=items_key,
            partition_attrs=partition_attrs,
            overlay_color_attr=overlay_color_attr,
            overlay_size_attr=overlay_size_attr,
            filename_prefix=filename_prefix,
            cluster_labels=cluster_labels,
            **kwargs,
        )

             
    def plot_citation_network(
        self,
        size_dict: Optional[Dict[str, float]] = None,
        color_dict: Optional[Dict[str, float]] = None,
        label_dict: Optional[Dict[str, str]] = None,
        use_default_dict: bool = True,
        cmap: str = "viridis",
        arrow_size: int = 10,
        font_size: int = 9,
        node_size_factor: float = 100.0,
        edge_width: float = 0.6,
        layout: str = "kamada_kawai",
        main_path_color: str = "crimson",
        main_path_width: float = 2.8,
        main_path_style: str = "solid",
        filename: Optional[str] = None,
        plot_mode: str = "just path",        # {"network","with_path","all","just path"}
        main_path_shape: str = "descending", # {"descending","line","curve_up","curve_down"}
        path_slope: float = 0.25,
        label_left_gap: float = 0.35,
        max_label_chars: int = 48,
        dpi: int = 170,
        seed: int = 42,
    ) -> None:
        """
        Plot the citation network and/or its main path.
    
        This method can:
          - plot the full citation network,
          - plot the full network with the main path highlighted,
          - plot only the main path (compact, readable layout),
          - or do all of the above.
    
        Main-path plot features
        -----------------------
        - Nodes placed along a descending polyline (by default).
        - Labels placed to the *left* of nodes, right-aligned.
        - `adjustText` (if available) is used to de-overlap labels vertically.
        - Figure height and label font size adapt to the number of path nodes
          to avoid extremely crowded, unreadable outputs.
    
        Parameters
        ----------
        size_dict : dict[str, float], optional
            Mapping node -> size value. If None, node degree is used.
        color_dict : dict[str, float or color], optional
            Mapping node -> numeric value or color. If provided, values are
            passed directly to NetworkX; otherwise a neutral color is used.
        label_dict : dict[str, str], optional
            Mapping node -> label text. If None and ``use_default_dict`` is
            True, ``self.citation_main_path_label_dict`` is used when present.
            If still missing, labels are built from node attributes such as
            "Short Label", "Title", "Authors", "Year".
        use_default_dict : bool, default True
            Whether to fall back to ``self.citation_main_path_label_dict`` when
            no ``label_dict`` is supplied.
        cmap : str, default "viridis"
            Colormap name for numeric node colors.
        arrow_size : int, default 10
            Arrow size used when drawing directed edges.
        font_size : int, default 9
            Base font size for labels (may be shrunk on long paths).
        node_size_factor : float, default 100.0
            Global multiplicative factor for all node sizes (in percent).
        edge_width : float, default 0.6
            Width of non-highlighted edges.
        layout : {"kamada_kawai", "spring"}, default "kamada_kawai"
            Layout algorithm used for the full network.
        main_path_color : str, default "crimson"
            Color of the edges on the main path.
        main_path_width : float, default 2.8
            Edge width for main-path edges.
        main_path_style : str, default "solid"
            Line style for main-path edges.
        filename : str, optional
            Base filename (without extension). If relative, it is stored under
            ``self.res_folder / "plots"``. If None, "citation network" is used.
            Files are saved as PNG, SVG and PDF.
        plot_mode : {"network", "with_path", "all", "just path"}, default "just path"
            What to draw:
              - "network"   : only full network
              - "with_path" : full network with main path highlighted
              - "just path" : only the main path plot
              - "all"       : all of the above
        main_path_shape : {"descending","line","curve_up","curve_down"}, default "descending"
            Shape of the main-path polyline.
        path_slope : float, default 0.25
            Slope parameter for "line"/"curve_*" shapes.
        label_left_gap : float, default 0.35
            Horizontal gap (data units) between node and its label on the path plot.
        max_label_chars : int, default 48
            Maximum label length; longer labels are truncated with an ellipsis.
        dpi : int, default 170
            DPI used for the figures.
        seed : int, default 42
            Random seed for layouts that use randomness.
    
        Returns
        -------
        None
        """
        import os
        import math
        from typing import Dict, Optional
    
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    
        # ------------------------------------------------------------------
        # Basic objects and output path
        # ------------------------------------------------------------------
        G: nx.DiGraph = self.citation_network_documents
        if G is None or G.number_of_nodes() == 0:
            return
    
        path = list(self.citation_main_path or [])
        cmap_obj = getattr(self, "cmap", cmap)
    
        plots_dir = os.path.join(self.res_folder, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    
        if filename is None:
            base = os.path.join(plots_dir, "citation network")
        else:
            # Allow passing either a bare name or a full path
            if os.path.isabs(filename):
                base = os.path.splitext(filename)[0]
            else:
                base = os.path.join(plots_dir, os.path.splitext(filename)[0])
    
        # ------------------------------------------------------------------
        # Size handling
        # ------------------------------------------------------------------
        def _fallback_sizes(graph: nx.Graph) -> Dict[str, float]:
            """Use node degree as a simple size proxy."""
            return {n: float(d) for n, d in graph.degree()}
    
        def _scale_sizes(raw: Dict[str, float]) -> Dict[str, float]:
            """Range-compress and normalize node sizes to a pleasant pixel span."""
            vals = np.array([max(0.0, float(v)) for v in raw.values()], dtype=float)
            if vals.size == 0 or float(vals.max()) <= 0.0:
                return {k: 300.0 * (node_size_factor / 100.0) for k in raw}
    
            positive = vals[vals > 0.0]
            vmin = float(positive.min()) if positive.size else 1.0
            vmax = float(vals.max())
            ratio = vmax / max(vmin, 1e-12)
    
            if ratio >= 100.0:
                compressed = {k: float(math.log1p(max(v, 0.0))) for k, v in raw.items()}
            else:
                compressed = {k: float(max(v, 0.0)) for k, v in raw.items()}
    
            cvals = np.array(list(compressed.values()), dtype=float)
            cmin, cmax = float(cvals.min()), float(cvals.max())
            if cmax == cmin:
                norm = {k: 0.5 for k in compressed}
            else:
                norm = {k: (v - cmin) / (cmax - cmin) for k, v in compressed.items()}
    
            min_px, max_px = 140.0, 950.0
            scale = node_size_factor / 100.0
            return {
                k: (min_px + n * (max_px - min_px)) * scale
                for k, n in norm.items()
            }
    
        node_sizes = _scale_sizes(size_dict or _fallback_sizes(G))
    
        # ------------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------------
        if label_dict is None and use_default_dict:
            label_dict = getattr(self, "citation_main_path_label_dict", None)
    
        def _first_author_short(authors: str) -> Optional[str]:
            """Return 'Surname I.' from a Scopus-like author string."""
            if not authors:
                return None
            first = str(authors).split(";")[0].strip()
            if not first:
                return None
            parts = first.replace(",", " ").split()
            if not parts:
                return None
            surname = parts[0]
            initial = f"{parts[1][0]}." if len(parts) > 1 else ""
            return f"{surname} {initial}".strip()
    
        def _build_label(node) -> str:
            """Build a readable label for a node."""
            data = G.nodes[node]
    
            # 1) user-supplied label_dict
            if label_dict and node in label_dict and str(label_dict[node]).strip():
                text = str(label_dict[node]).strip()
            else:
                # 2) look for convenient attributes on the node
                for key in (
                    "Short Label",
                    "short_label",
                    "Label",
                    "label",
                    "Title",
                    "title",
                ):
                    value = data.get(key)
                    if value and str(value).strip():
                        text = str(value).strip()
                        break
                else:
                    # 3) fall back to "FirstAuthor et al. (Year)" or node id
                    authors = data.get("Authors") or data.get("authors")
                    year = (
                        data.get("Year")
                        or data.get("publication_year")
                        or data.get("year")
                    )
                    fa = _first_author_short(authors) if authors else None
                    year_text = ""
                    if year not in (None, ""):
                        try:
                            year_int = int(float(str(year)))
                            year_text = f" ({year_int})"
                        except Exception:
                            pass
                    if fa:
                        text = f"{fa} et al.{year_text}"
                    else:
                        text = str(node)
    
            # normalize whitespace and truncate
            text = " ".join(str(text).split())
            if len(text) > max_label_chars:
                text = text[: max_label_chars - 1].rstrip() + "…"
            return text
    
        labels = {n: _build_label(n) for n in G.nodes()}
    
        # ------------------------------------------------------------------
        # Colors
        # ------------------------------------------------------------------
        def _node_colors(nodes):
            """Return either a list from color_dict or a neutral color."""
            if color_dict and any(color_dict.get(n) is not None for n in nodes):
                return [color_dict.get(n) for n in nodes]
            return "lightgray"
    
        # ------------------------------------------------------------------
        # Full network drawing
        # ------------------------------------------------------------------
        def _draw_full_network(highlight_path: bool) -> None:
            """Draw the full citation network, optionally highlighting main path."""
            n_nodes = G.number_of_nodes()
    
            # Dynamic figure size: larger graphs get a bit more space
            fig_w = max(8.0, min(16.0, 8.0 + 0.04 * max(n_nodes - 20, 0)))
            fig_h = max(5.0, min(9.0, 5.0 + 0.04 * max(n_nodes - 20, 0)))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    
            if layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G, seed=seed)
    
            norder = list(G.nodes())
            ns = [node_sizes.get(n, 300.0) for n in norder]
            nc = _node_colors(norder)
    
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=ns,
                node_color=nc,
                cmap=cmap_obj,
                edgecolors="white",
                linewidths=0.8,
                ax=ax,
            )
    
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_width,
                arrows=True,
                arrowsize=arrow_size,
                alpha=0.55,
                ax=ax,
            )
    
            if highlight_path and len(path) >= 2:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(zip(path[:-1], path[1:])),
                    width=main_path_width,
                    edge_color=main_path_color,
                    style=main_path_style,
                    arrows=True,
                    arrowsize=arrow_size * 1.15,
                    ax=ax,
                )
    
            # Label a reasonable subset if graph is large
            if n_nodes <= 40:
                label_nodes = norder
            else:
                label_nodes = list(dict.fromkeys(path + norder[:40]))  # preserve order
    
            label_pos = {n: (pos[n][0], pos[n][1] + 0.02) for n in label_nodes}
            nx.draw_networkx_labels(
                G,
                label_pos,
                labels={n: labels[n] for n in label_nodes},
                font_size=font_size,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.82,
                ),
                clip_on=False,
                ax=ax,
            )
    
            ax.set_axis_off()
            fig.tight_layout()
    
            suffix = " with main path" if highlight_path else ""
            out_base = f"{base}{suffix}"
            for ext in ("png", "svg", "pdf"):
                fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
            plt.close(fig)
    
        # ------------------------------------------------------------------
        # Main path drawing
        # ------------------------------------------------------------------
        def _draw_main_path() -> None:
            """Draw only the main citation path with de-overlapped labels."""
            if not path:
                return
    
            H = nx.DiGraph()
            H.add_nodes_from(path)
            H.add_edges_from(list(zip(path[:-1], path[1:])))
    
            n_path = len(path)
            x = np.arange(n_path, dtype=float)
    
            # y-shape of the path
            if main_path_shape == "descending":
                # enforce clear vertical spacing
                y = -x
            elif main_path_shape == "line":
                slope = path_slope if path_slope != 0 else -0.4
                y = slope * (x - x.mean())
            elif main_path_shape == "curve_up":
                y = 0.25 * np.log1p(x)
            elif main_path_shape == "curve_down":
                y = -0.25 * np.log1p(x)
            else:
                y = -x
    
            pos = {n: (float(xi), float(yi)) for n, xi, yi in zip(path, x, y)}
    
            # Dynamic figure size: more nodes => taller and somewhat wider figure
            fig_h = max(3.4, min(10.0, 3.4 + 0.18 * max(n_path - 10, 0)))
            fig_w = max(6.5, min(18.0, 0.6 * n_path + 4.0))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    
            # Node sizes & colors, slightly clamped
            ns = [max(120.0, min(950.0, node_sizes.get(n, 300.0))) for n in path]
            nc = _node_colors(path)
    
            nx.draw_networkx_nodes(
                H,
                pos,
                node_size=ns,
                node_color=nc,
                cmap=cmap_obj,
                edgecolors="white",
                linewidths=0.8,
                ax=ax,
            )
    
            nx.draw_networkx_edges(
                H,
                pos,
                edgelist=list(zip(path[:-1], path[1:])),
                width=main_path_width,
                edge_color=main_path_color,
                style=main_path_style,
                arrows=True,
                arrowsize=arrow_size * 1.15,
                connectionstyle="arc3,rad=0.0",
                ax=ax,
            )
    
            # Effective font size: shrink a bit for very long paths
            eff_font_size = font_size
            if n_path > 25:
                eff_font_size = max(6, font_size - 0.15 * (n_path - 25))
    
            # Create left-of-node texts
            texts = []
            fixed_x = []
            for node in path:
                xx, yy = pos[node]
                t = ax.text(
                    xx - label_left_gap,
                    yy,
                    labels[node],
                    fontsize=eff_font_size,
                    color="black",
                    ha="right",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="none",
                        alpha=0.92,
                    ),
                    clip_on=False,
                )
                t.set_path_effects(
                    [pe.Stroke(linewidth=2, foreground="white"), pe.Normal()]
                )
                texts.append(t)
                fixed_x.append(xx - label_left_gap)
    
            # Use adjustText if available to de-overlap vertically
            try:
                from adjustText import adjust_text
    
                for _ in range(5):
                    adjust_text(
                        texts,
                        ax=ax,
                        expand_text=(1.02, 1.12),
                        expand_points=(1.02, 1.12),
                        force_text=(0.02, 0.15),
                        lim=5000,
                        avoid_points=True,
                        only_move={"points": "y", "text": "y"},
                        autoalign=False,
                    )
                    # Re-pin x so labels stay exactly left of nodes
                    for t, x0 in zip(texts, fixed_x):
                        y_now = t.get_position()[1]
                        t.set_position((x0, y_now))
            except Exception:
                # Fallback: simple vertical staggering
                for i, t in enumerate(texts):
                    x0, y0 = t.get_position()
                    offset = 0.12 * ((i % 3) - 1)  # -0.12, 0, +0.12
                    t.set_position((x0, y0 + offset))
    
            ax.set_axis_off()
            xmin, xmax = x.min() - 0.6 - label_left_gap, x.max() + 0.6
            ymin, ymax = y.min() - 1.0, y.max() + 1.0
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    
            out_base = f"{base} main path"
            for ext in ("png", "svg", "pdf"):
                fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
            plt.close(fig)
    
        # ------------------------------------------------------------------
        # Dispatch according to plot_mode
        # ------------------------------------------------------------------
        mode = (plot_mode or "just path").strip().lower().replace("_", " ")
        if mode in {"network", "all"}:
            _draw_full_network(highlight_path=False)
        if mode in {"with path", "with_path", "all"}:
            _draw_full_network(highlight_path=True)
        if mode in {"just path", "path", "all"}:
            _draw_main_path()




    def plot_k_field_sankey(
        self,
        fields: list[str] = ("author keywords", "sources", "authors"),
        customs: dict[str, pd.DataFrame] | None = None,
        top_n: int | list[int] = 10,
        color_option: str = "Average year",
        label_maps: dict[str, str] | None = None,
        colorscale: str | list | None = None,
        save_png: str | None = None,
        save_html: str | None = None,
    ):
        """
        Build and render a k-field Sankey diagram mixing known and custom concepts.
    
        Changes in this version
        -----------------------
        - If `colorscale` is not provided, falls back to `self.cmap` (else "Viridis").
        - If `self.res_folder` is set and `save_png` is not provided, the plot is
          saved automatically under `<self.res_folder>/plots/` with a descriptive
          filename based on the chosen fields and color metric.
    
        Parameters
        ----------
        fields : list[str], default ("author keywords", "sources", "authors")
            Ordered field names to include (known from global `mapping` or custom).
        customs : dict[str, pd.DataFrame], optional
            {custom_name -> docs×concepts binary DataFrame}. Overrides known lookups.
        top_n : int or list[int], default 10
            Top columns kept per field (by column sum). If a list, applied per field.
        color_option : {"Average year", "Citations per document"} or str
            Coloring metric taken from `self.df`. Unknown/missing → no colors.
        label_maps : dict[str, str], optional
            Display-only label remapping applied after topology is built.
        colorscale : str | list | None, default None
            Plotly colorscale; if None, uses `self.cmap` or "Viridis".
        save_png : str | None, default None
            PNG path. If None and `self.res_folder` exists, a descriptive name is created.
        save_html : str | None, default None
            HTML path (disabled by default).
    
        Returns
        -------
        plotly.graph_objects.Figure
            The Sankey figure.
        """
        import os, re
        from pathlib import Path
    
        # ---------- helpers ----------
        def _slug(s: str) -> str:
            s = str(s)
            return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    
        # Use self.cmap when user didn’t pass a colorscale
        if colorscale is None:
            colorscale = getattr(self, "cmap", "Viridis")
    
        # access global mapping (dict of dicts of strings → self attribute/method names)
        try:
            mp = self.mapping
        except KeyError as e:
            raise RuntimeError('Global variable "mapping" is not imported in this module.') from e
        if not isinstance(mp, dict):
            raise TypeError('Global "mapping" must be a dict-of-dicts of strings.')
    
        def _norm(s: str) -> str:
            return " ".join(str(s).strip().lower().replace("’", "'").split())
    
        aliases: dict[str, set[str]] = {
            "author keywords": {"author keywords", "authors keywords", "author's keywords", "keywords"},
            "sources": {"sources", "journals", "venues", "source title", "source titles"},
            "authors": {"authors", "author", "author full names", "author(s)"},
        }
        norm_to_real = {_norm(k): k for k in mp.keys()}
    
        def _resolve_key(field_name: str) -> str:
            n = _norm(field_name)
            for canonical, alias_set in aliases.items():
                if n in {_norm(a) for a in alias_set}:
                    n = _norm(canonical); break
            real = norm_to_real.get(n)
            if real is None:
                avail = ", ".join(sorted(mp.keys()))
                raise KeyError(f'Field "{field_name}" not found in global mapping. Available: {avail}')
            return real
    
        def _ensure_binary_for_known(field_name: str, per_field_top_n: int) -> pd.DataFrame:
            key = _resolve_key(field_name)
            entry = mp.get(key, {})
            if not isinstance(entry, dict):
                raise TypeError(f"mapping[{key!r}] must be a dict.")
            bin_attr = entry.get("binary indicators")
            if not isinstance(bin_attr, str):
                raise KeyError(f'mapping["{key}"]["binary indicators"] must be a string attribute name.')
            df_bin = getattr(self, bin_attr, None)
    
            def _needs_build(x) -> bool:
                return not isinstance(x, pd.DataFrame) or x.empty or x.shape[1] == 0
    
            if _needs_build(df_bin):
                counter_attr = entry.get("counter")
                if not isinstance(counter_attr, str):
                    raise KeyError(f'Binary for "{key}" missing and mapping["{key}"]["counter"] not defined.')
                if not hasattr(self, counter_attr):
                    raise AttributeError(f'self has no method "{counter_attr}" referenced by mapping["{key}"]["counter"].')
                counter_fn = getattr(self, counter_attr)
                try:
                    counter_fn(top_n=int(per_field_top_n))
                except TypeError:
                    counter_fn()  # fallback if counter has no top_n
                df_bin = getattr(self, bin_attr, None)
                if _needs_build(df_bin):
                    raise RuntimeError(f'After calling counter "{counter_attr}", attribute "{bin_attr}" is still empty.')
            if not isinstance(df_bin, pd.DataFrame):
                raise TypeError(f'Attribute "{bin_attr}" must be a pandas DataFrame.')
            return df_bin
    
        # ---------- collect matrices ----------
        customs = {} if customs is None else customs
        if isinstance(top_n, int):
            top_n_list = [top_n] * len(fields)
        else:
            top_n_list = list(top_n)
            if len(top_n_list) < len(fields):
                top_n_list += [top_n_list[-1]] * (len(fields) - len(top_n_list))
            elif len(top_n_list) > len(fields):
                top_n_list = top_n_list[:len(fields)]
    
        dfs: list[pd.DataFrame] = []
        field_names: list[str] = []
        for fld, n_this in zip(fields, top_n_list):
            if fld in customs:
                df_bin = customs[fld]
                if not isinstance(df_bin, pd.DataFrame):
                    raise TypeError(f"customs[{fld!r}] must be a pandas DataFrame.")
            else:
                df_bin = _ensure_binary_for_known(fld, int(n_this))
    
            df_bin = (
                df_bin.reindex(self.df.index)
                      .fillna(0)
                      .astype(bool)
                      .astype("uint8")
            )
            dfs.append(df_bin)
            field_names.append(str(fld))
    
        # ---------- colors ----------
        color_series: pd.Series | None = None
        color_func = np.mean
        cmap = {"Average year": ("Year", np.mean), "Citations per document": ("Cited by", np.mean)}
        if color_option in cmap:
            col, color_func = cmap[color_option]
            if col in self.df.columns:
                color_series = self.df[col]
    
        # ---------- prepare + plot ----------
        links_df, labels, color_values, _, group_ids = plotbib.prepare_for_sankey(
            dataframes=dfs,
            top_n=top_n_list,
            label_maps=label_maps,
            color_series=color_series,
            color_func=color_func,
            all_pairs=False,
        )
    
        # auto filename if res_folder is set and user didn't pass save_png
        if save_png is None and getattr(self, "res_folder", None):
            plots_dir = Path(self.res_folder) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fields_part = "-".join(_slug(f) for f in field_names)
            color_part = f"_by-{_slug(color_option)}" if color_series is not None else ""
            save_png = str(plots_dir / f"sankey_{fields_part}{color_part}.png")
    
        fig = plotbib.plot_sankey(
            links_df=links_df,
            labels=labels,
            color_values=(color_values if color_series is not None else None),
            group_ids=group_ids,
            field_names=field_names,
            save_png=save_png,
            save_html=save_html,                 # HTML still off unless provided
            colorscale=colorscale,               # uses self.cmap if user didn't pass
            colorbar_title=(color_option if color_series is not None else ""),
        )
        return fig


    def plot_country_collaboration(self, top_n_pairs=20, connect_threshold=1, top_n_countries=20, annotate_heatmap=True, figsizes={"pairs": (10,6), "network": (12,12), "heatmap": (12,10)}, filename="country collaboration", **kwargs):
        
        """Plot several views of international collaboration between countries.

        Depending on the chosen ``kind`` this can produce bar charts for the
        top collaborating pairs, a collaboration network or heatmaps of the
        collaboration matrix.

        Parameters
        ----------
        top_n_pairs : int, default 20
            Number of country pairs to highlight in pair-based plots.
        figsize_dict : dict, optional
            Optional mapping from plot kind to figure size.
        filename : str, default "country collaboration"
            Base filename (without extension) for saving the plots into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to the underlying plotting
            routines in :mod:`plotbib`.
        """
        filename = os.path.join(self.res_folder, "plots", filename)
        plotbib.plot_top_country_pairs(self.country_collab_matrix, top_n=top_n_pairs, figsize=figsizes["pairs"], filename_base=filename + "top pairs")
        plotbib.plot_country_collab_network(self.country_collab_matrix, threshold=connect_threshold, figsize=figsizes["network"], layout_func="spring", filename_base=filename + "network")
        plotbib.plot_country_collab_heatmap(self.country_collab_matrix, top_n=top_n_countries, figsize=figsizes["heatmap"], cmap=self.cmap, annotate=annotate_heatmap, filename_base=filename + "heatmap")
        plotbib.plot_country_collab_heatmap(self.country_collab_matrix_norm, top_n=top_n_countries, figsize=figsizes["heatmap"], cmap=self.cmap, annotate=annotate_heatmap, filename_base=filename + "heatmap normalized")
        #self.plot_coocurence_network("all countries", **kwargs)
                    
    def plot_historiograph(self, figsize=(12, 8), size_attr=None, min_indegree=None,
                              min_citations=100, min_year=None, max_year=None, filename="historiograph", **kwargs):
        """Plot a historiograph (main path) of the citation network.

        If the historiograph network has not been built yet it is created via
        :meth:`build_historiograph`. The layout is computed with
        :func:`plotbib.layout_historiograph` and the final visualisation with
        :func:`plotbib.plot_historiograph`.

        Parameters
        ----------
        figsize : tuple of float, default (12, 8)
            Size of the figure.
        size_attr : str or None, optional
            Node attribute used to scale node sizes. If None, a default measure
            is used.
        min_indegree : int or None, optional
            Minimum in-degree required for a node to be shown.
        min_citations : int, default 100
            Minimum number of citations required for a node to be shown.
        min_year : int or None, optional
            Minimum publication year to include.
        max_year : int or None, optional
            Maximum publication year to include.
        filename : str, default "historiograph"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_historiograph`.
        """
        if not hasattr(self, "historiograph"):
            self.build_historiograph(**kwargs)
        G = self.historiograph
        pos = plotbib.layout_historiograph(G)
        
        filename = os.path.join(self.res_folder, "plots", filename)
        plotbib.plot_historiograph(G, pos, figsize=figsize, size_attr=size_attr,
                                   min_indegree=min_indegree, min_citations=min_citations,
                                   min_year=min_year, max_year=max_year, save_as=filename,
                                   dpi=self.dpi)
        
    # plotting of relations

    # ---------- some helpers -----------------

    def _safe_name(self, s: str) -> str:
        """
        Make a string filesystem-safe by replacing spaces and stripping unsafe chars.
        """
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(s).strip().replace(" ", "_"))
    
    def _relations_dir(self) -> str | None:
        """
        Ensure and return "<self.res_folder>/relations", or None if res_folder is unset.
        """
        base = getattr(self, "res_folder", None)
        if not base:
            return None
        out = os.path.join(base, "relations")
        os.makedirs(out, exist_ok=True)
        return out
    
    def _default_base(self, c1: str, c2: str, tag: str) -> str | None:
        """
        Build "<res_folder>/relations/<tag>__<c1>__<c2>" (no extension) or None if res_folder is unset.
        """
        d = self._relations_dir()
        if not d:
            return None
        return os.path.join(d, f"{self._safe_name(tag)}__{self._safe_name(c1)}__{self._safe_name(c2)}")
    
    
    # ---------- wrappers that call your plotbib.* implementations -----------------
    
    def plot_relation_correspondence(
        self,
        concept1: str,
        concept2: str,
        *,
        filename_base: str | None = None,
        **kwargs,
    ):
        """
        Plot correspondence analysis for (concept1, concept2) using plotbib.plot_correspondence_analysis.
    
        Behavior
        --------
        - If the relation already exists in `self.relations[concept1][concept2]`, reuse it.
          If not present, compute it via `self.relate_concepts(..., include_stats=("correspondence",))`.
        - If an existing relation is found but lacks correspondence stats, it is (re)computed
          with `include_stats=("correspondence",)` to ensure plotting works.
        - DPI is taken from `self.dpi` (fallback 600); any incoming "dpi" in kwargs is ignored.
        - The default save location (when `filename_base` is None) is `{self.res_folder}/relations/`.
    
        Parameters
        ----------
        concept1, concept2 : str
            Concepts defining the relation.
        filename_base : str | None
            Base path (no extension). If None and `self.res_folder` is set, a default inside
            the "relations" subfolder is used.
        **kwargs
            Forwarded to `plotbib.plot_correspondence_analysis`
            (e.g., "figsize", "annotate", "alpha", "size_scale", "use_size",
            "row_label_name", "col_label_name", "title", "abbreviate_labels", "abbreviate_kwargs").
    
        Returns
        -------
        None
    
        Examples
        --------
        # one-liners
        ba.plot_relation_correspondence("Author Keywords", "Index Keywords", title="CA: AK vs IK")
        ba.plot_relation_correspondence("A", "B", figsize=(9, 7), annotate=True, size_scale=50)
        """
        import os
        from biblium import plotbib
    
        # Enforce DPI from self and ignore any incoming "dpi"
        kwargs.pop("dpi", None)
        eff_dpi = getattr(self, "dpi", 600)
    
        # Try to reuse an existing relation from the double-dict store
        R = None
        rels = getattr(self, "relations", {}) or {}
        if isinstance(rels, dict) and concept1 in rels and isinstance(rels[concept1], dict):
            R = rels[concept1].get(concept2, None)
    
        # If not found under (concept1, concept2), optionally check reverse for robustness
        if R is None and isinstance(rels, dict) and concept2 in rels and isinstance(rels[concept2], dict):
            R = rels[concept2].get(concept1, None)
    
        # Ensure correspondence stats; compute/recompute when needed
        def _has_ca_stats(obj) -> bool:
            return (
                hasattr(obj, "ca_row_coords")
                and hasattr(obj, "ca_col_coords")
                and getattr(obj, "ca_row_coords") is not None
                and getattr(obj, "ca_col_coords") is not None
            )
    
        if R is None or not _has_ca_stats(R):
            R = self.relate_concepts(concept1, concept2, include_stats=("correspondence",))
    
        # Extract CA components
        rc = getattr(R, "ca_row_coords", None)
        cc = getattr(R, "ca_col_coords", None)
        inertia = list(getattr(R, "ca_explained_inertia", []) or [])  # default to empty -> handled below
        if not inertia:
            inertia = [0.0, 0.0]
    
        # Default filename base inside {res_folder}/relations if not provided
        if filename_base is None and getattr(self, "res_folder", None):
            safe_c1 = str(concept1).replace(os.sep, "_")
            safe_c2 = str(concept2).replace(os.sep, "_")
            filename_base = os.path.join(self.res_folder, "relations", f"{safe_c1}__{safe_c2}__CA")
    
        # Dispatch to plotter
        plotbib.plot_correspondence_analysis(
            row_coords=rc,
            col_coords=cc,
            explained_inertia=inertia,
            df_relation=getattr(R, "rm", None),
            filename_base=filename_base,
            dpi=eff_dpi,
            **kwargs,
        )
    
    def plot_relation_residual_heatmap(
        self,
        concept1: str,
        concept2: str,
        *,
        filename_base: str | None = None,
        **kwargs,
    ):
        """
        Plot Pearson standardized residuals heatmap via plotbib.plot_residual_heatmap.
    
        Behavior
        --------
        - Reuses an existing relation in self.relations[concept1][concept2] if available;
          otherwise computes it. If present but missing chi2 stats, recomputes with
          include_stats=("chi2",).
        - DPI forced from self.dpi (fallback 600). Colormap forced from self.cmap if present.
        - Default save goes to {self.res_folder}/relations when filename_base is None.
    
        Parameters
        ----------
        concept1, concept2 : str
            Concepts defining the relation.
        filename_base : str | None
            Base path (no extension). If None and self.res_folder is set, a default is used.
        **kwargs
            Forwarded to plotbib.plot_residual_heatmap (e.g., figsize, annotate, square, center, title,
            row_label, col_label, and any seaborn heatmap extras).
    
        Examples
        --------
        ba.plot_relation_residual_heatmap("Author Keywords", "Index Keywords", figsize=(9, 7), annotate=True, square=True, title="Residuals")
        """
        import os
        import pandas as pd
        from biblium import plotbib
    
        # enforce DPI/cmap from self
        kwargs.pop("dpi", None)
        kwargs.pop("cmap", None)
        eff_dpi = getattr(self, "dpi", 600)
        eff_cmap = getattr(self, "cmap", None)
    
        # Try reuse relation from store (check both orders)
        R = None
        rels = getattr(self, "relations", {}) or {}
        if isinstance(rels, dict) and concept1 in rels and isinstance(rels[concept1], dict):
            R = rels[concept1].get(concept2)
        if R is None and isinstance(rels, dict) and concept2 in rels and isinstance(rels[concept2], dict):
            R = rels[concept2].get(concept1)
    
        # Ensure chi2 stats
        need_stats = True
        if R is not None and hasattr(R, "chi2_residuals_df"):
            need_stats = not isinstance(R.chi2_residuals_df, pd.DataFrame)
        if R is None or need_stats:
            R = self.relate_concepts(concept1, concept2, include_stats=("chi2",))
    
        if R.chi2_residuals_df is None or not isinstance(R.chi2_residuals_df, pd.DataFrame):
            raise ValueError("Chi-square residuals are unavailable for this relation.")
    
        # Default filename base inside relations subfolder
        if filename_base is None and getattr(self, "res_folder", None):
            safe_c1 = str(concept1).replace(os.sep, "_")
            safe_c2 = str(concept2).replace(os.sep, "_")
            filename_base = os.path.join(self.res_folder, "relations", f"{safe_c1}__{safe_c2}__residuals")
    
        plotbib.plot_residual_heatmap(
            residuals_df=R.chi2_residuals_df,
            filename_base=filename_base,
            dpi=eff_dpi,
            cmap=eff_cmap,
            **kwargs,
        )
    
    
    def plot_relation_bipartite_network(
        self,
        concept1: str,
        concept2: str,
        *,
        filename_base: str | None = None,
        **kwargs,
    ):
        """
        Plot bipartite graph via plotbib.plot_bipartite_network.
    
        Behavior
        --------
        - Reuses an existing relation from the store when available; otherwise computes it.
          If present but missing the bipartite graph, recomputes with include_stats=("bipartite network",).
        - DPI forced from self.dpi (fallback 600).
        - Default save goes to {self.res_folder}/relations when filename_base is None.
    
        Parameters
        ----------
        concept1, concept2 : str
            Concepts defining the relation.
        filename_base : str | None
            Base path (no extension). If None and self.res_folder is set, a default is used.
        **kwargs
            Forwarded to plotbib.plot_bipartite_network (e.g., node_size_scale, edge_alpha, same_size,
            weight_threshold, show_edge_weights, edge_width_scale, figsize, title,
            row_label_name, col_label_name).
    
        Examples
        --------
        ba.plot_relation_bipartite_network("Author Keywords", "Index Keywords", node_size_scale=0.8, edge_alpha=0.3, title="Bipartite")
        """
        import os
        from biblium import plotbib
    
        # enforce DPI from self
        kwargs.pop("dpi", None)
        eff_dpi = getattr(self, "dpi", 600)
    
        # Try reuse relation (check both orders)
        R = None
        rels = getattr(self, "relations", {}) or {}
        if isinstance(rels, dict) and concept1 in rels and isinstance(rels[concept1], dict):
            R = rels[concept1].get(concept2)
        if R is None and isinstance(rels, dict) and concept2 in rels and isinstance(rels[concept2], dict):
            R = rels[concept2].get(concept1)
    
        # Ensure bipartite network stats
        if R is None or getattr(R, "bipartite_graph", None) is None:
            R = self.relate_concepts(concept1, concept2, include_stats=("bipartite network",))
    
        if getattr(R, "bipartite_graph", None) is None:
            raise ValueError("Bipartite graph is unavailable for this relation.")
    
        G = R.bipartite_graph
        graph_nodes = set(G.nodes())
        row_nodes = [n for n in R.rm.index if n in graph_nodes]
        col_nodes = [n for n in R.rm.columns if n in graph_nodes]
    
        # Default filename base
        if filename_base is None and getattr(self, "res_folder", None):
            safe_c1 = str(concept1).replace(os.sep, "_")
            safe_c2 = str(concept2).replace(os.sep, "_")
            filename_base = os.path.join(self.res_folder, "relations", f"{safe_c1}__{safe_c2}__bipartite")
    
        plotbib.plot_bipartite_network(
            B=G,
            row_nodes=row_nodes,
            col_nodes=col_nodes,
            filename_base=filename_base,
            dpi=eff_dpi,
            **kwargs,
        )
    
    
    def plot_relation_top_n_pairs(
        self,
        concept1: str,
        concept2: str,
        *,
        source: str = "chi2",                 # "chi2" or "log_ratio"
        metric_column: str | None = None,
        filename_base: str | None = None,
        order: str = "freq",                  # "freq" (default), "alpha", or "custom"
        row_order: list[str] | None = None,   # used when order="custom"
        col_order: list[str] | None = None,   # used when order="custom"
        **kwargs,
    ):
        """
        Plot top-N row/column pairs via plotbib.plot_top_n_pairs with configurable axis ordering
        and **uniform, small bubble size**.
    
        Size behavior
        -------------
        - Any incoming "size_column" is ignored and removed.
        - Any incoming "size_scale" is overridden.
        - Uniform bubble size is taken from `self.pair_bubble_size` if present, else 2.0.
    
        Ordering
        --------
        - order="freq": axes ordered by marginal frequency (row/col totals, descending),
          taken from the relation's contingency table; falls back to sums of "Count"/|metric|.
        - order="alpha": alphabetical order.
        - order="custom": use provided `row_order`/`col_order` (labels not present are ignored).
    
        Other behavior
        --------------
        - Reuses existing relation; if χ² residuals are missing for custom matrices, computes them
          from R's contingency; otherwise falls back to `self.relate_concepts(..., include_stats=("chi2",))`.
        - DPI enforced from `self.dpi`; colormap (plotbib arg "color_map") from `self.cmap`.
        - Default save path: `{self.res_folder}/relations/...` when `filename_base` is None.
    
        Examples (one-liners)
        ---------------------
        # Default (freq ordering, small uniform bubbles)
        # ba.plot_relation_top_n_pairs("Author Keywords", "Index Keywords", source="chi2", top_n=30, title="Top χ² pairs")
    
        # Alphabetical
        # ba.plot_relation_top_n_pairs("A", "B", source="log_ratio", order="alpha", top_n=20, title="Top log-ratio pairs")
    
        # Custom order
        # ba.plot_relation_top_n_pairs("A", "B", source="chi2", order="custom", row_order=["x3","x1","x2"], top_n=25, title="Custom row order")
        """
        import os
        import numpy as np
        import pandas as pd
        from biblium import plotbib
    
        # Enforce DPI/CMAP from self
        kwargs.pop("dpi", None)
        kwargs.pop("cmap", None)
        kwargs.pop("color_map", None)
        eff_dpi = getattr(self, "dpi", 600)
        eff_cmap = getattr(self, "cmap", None)
    
        # >>> Use proportional bubble sizes based on observed counts <<<
        # Allow size_column to be passed; default to "observed" for proportional sizes
        if "size_column" not in kwargs:
            kwargs["size_column"] = "observed"  # Proportional to intersection count
        # Set reasonable size range if not specified
        if "min_size" not in kwargs:
            kwargs["min_size"] = 50.0
        if "max_size" not in kwargs:
            kwargs["max_size"] = 500.0
    
        if source not in {"chi2", "log_ratio"}:
            raise ValueError("source must be \"chi2\" or \"log_ratio\".")
        if order not in {"freq", "alpha", "custom"}:
            raise ValueError("order must be \"freq\", \"alpha\", or \"custom\".")
        if order == "custom" and (row_order is None and col_order is None):
            raise ValueError("when order=\"custom\", provide row_order and/or col_order.")
    
        # Locate an existing relation (check both directions)
        R = None
        rels = getattr(self, "relations", {}) or {}
        if isinstance(rels, dict) and concept1 in rels and isinstance(rels[concept1], dict):
            R = rels[concept1].get(concept2)
        if R is None and isinstance(rels, dict) and concept2 in rels and isinstance(rels[concept2], dict):
            R = rels[concept2].get(concept1)
    
        # Helpers
        def _find_contingency_df(obj) -> pd.DataFrame | None:
            if obj is None:
                return None
            for name in ("matrix", "contingency", "contingency_table", "observed", "counts", "table", "M", "df"):
                cand = getattr(obj, name, None)
                if isinstance(cand, pd.DataFrame) and cand.shape[0] > 0 and cand.shape[1] > 0:
                    return cand
            for v in getattr(obj, "__dict__", {}).values():
                if isinstance(v, pd.DataFrame) and v.shape[0] > 0 and v.shape[1] > 0:
                    return v
            return None
    
        def _compute_chi2_sorted_pairs_from_df(df: pd.DataFrame) -> pd.DataFrame:
            obs = df.astype(float).copy()
            total = float(obs.values.sum())
            if total <= 0:
                raise ValueError("Contingency table is empty; cannot compute χ² residuals.")
            row_sum = obs.sum(axis=1).values.reshape(-1, 1)
            col_sum = obs.sum(axis=0).values.reshape(1, -1)
            expected = (row_sum @ col_sum) / total
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = (obs.values - expected) / np.sqrt(expected)
                resid = np.where(expected == 0.0, 0.0, resid)
            rows = np.repeat(obs.index.values, obs.shape[1])
            cols = np.tile(obs.columns.values, obs.shape[0])
            pairs = pd.DataFrame(
                {"Row": rows, "Column": cols, "Count": obs.values.ravel(order="C"),
                 "Expected": expected.ravel(order="C"), "Residual": resid.ravel(order="C")}
            )
            pairs["abs_metric"] = pairs["Residual"].abs()
            pairs.sort_values("abs_metric", ascending=False, inplace=True, kind="mergesort")
            pairs.reset_index(drop=True, inplace=True)
            return pairs
    
        # Ensure stats exist but preserve custom matrices for chi2 path
        if source == "chi2":
            have_df = isinstance(getattr(R, "chi2_sorted_residuals", None), pd.DataFrame)
            if not have_df:
                cont = _find_contingency_df(R)
                if isinstance(cont, pd.DataFrame):
                    try:
                        R.chi2_sorted_residuals = _compute_chi2_sorted_pairs_from_df(cont)
                    except Exception:
                        R = None
                else:
                    R = None
            if R is None:
                R = self.relate_concepts(concept1, concept2, include_stats=("chi2",))
        else:
            if R is None or not isinstance(getattr(R, "log_ratio_sorted_log_ratios", None), pd.DataFrame):
                R = self.relate_concepts(concept1, concept2, include_stats=("log-ratio",))
    
        # Pick pairs and defaults
        if source == "chi2":
            df_pairs = R.chi2_sorted_residuals
            mcol = metric_column or "Residual"
            tag = "top_pairs_chi2"
        else:
            df_pairs = R.log_ratio_sorted_log_ratios
            mcol = metric_column or "LogRatio"
            tag = "top_pairs_logratio"
    
        if df_pairs is None or not isinstance(df_pairs, pd.DataFrame):
            raise ValueError(f"Sorted pairs are unavailable for source \"{source}\".")
    
        # Build axis orders
        row_order_final = None
        col_order_final = None
        if order == "freq":
            cont = _find_contingency_df(R)
            if isinstance(cont, pd.DataFrame):
                row_tot = cont.sum(axis=1).sort_values(ascending=False)
                col_tot = cont.sum(axis=0).sort_values(ascending=False)
                row_order_final = list(row_tot.index.astype(str))
                col_order_final = list(col_tot.index.astype(str))
            else:
                if "Count" in df_pairs.columns:
                    row_order_final = list(df_pairs.groupby("Row")["Count"].sum().sort_values(ascending=False).index.astype(str))
                    col_order_final = list(df_pairs.groupby("Column")["Count"].sum().sort_values(ascending=False).index.astype(str))
        elif order == "custom":
            row_order_final = [str(x) for x in row_order] if row_order is not None else None
            col_order_final = [str(x) for x in col_order] if col_order is not None else None
        # alpha => None -> alphabetical in plotter
    
        # Default filename inside relations/
        if filename_base is None and getattr(self, "res_folder", None):
            safe_c1 = str(concept1).replace(os.sep, "_")
            safe_c2 = str(concept2).replace(os.sep, "_")
            filename_base = os.path.join(self.res_folder, "relations", f"{safe_c1}__{safe_c2}__{tag}")
        if filename_base:
            os.makedirs(os.path.dirname(filename_base), exist_ok=True)
    
        # Delegate to plotter (which already uses uniform small bubbles)
        plotbib.plot_top_n_pairs(
            sorted_pairs_df=df_pairs,
            metric_column=mcol,
            filename_base=filename_base,
            dpi=eff_dpi,
            color_map=eff_cmap,
            row_order=row_order_final,
            col_order=col_order_final,
            **kwargs,
        )

    def plot_keyword_bursts(self, 
                          keyword_col="Processed Author Keywords", 
                          top_n=30, 
                          s=2.0, 
                          gamma=1.0,
                          filename="keyword_bursts"):
        """
        Visualizes keyword bursts (Kleinberg's algorithm).
        """
        # Compute bursts
        burst_df = self.compute_bursts(keyword_col=keyword_col, top_n=top_n, s=s, gamma=gamma)
        
        if burst_df.empty:
            logging.warning("No bursts to plot.")
            return
            
        # Sort for plotting: Earliest start time first, then by weight
        burst_df = burst_df.sort_values(by=["Start", "Weight"], ascending=[True, False])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, len(burst_df) * 0.4 + 2))
        
        # Create a timeline for each keyword
        # Y-axis will be the keywords, X-axis is Year
        
        # Get unique keywords in order
        keywords = burst_df["Keyword"].unique()
        y_pos = range(len(keywords))
        kw_map = {kw: i for i, kw in enumerate(keywords)}
        
        # Draw the lines
        # First, draw a thin grey line for the full duration of the dataset for context? 
        # Or just the bursts. Let's do just the bursts.
        
        min_year = self.df["Year"].min()
        max_year = self.df["Year"].max()
        
        # Grid and Bounds
        ax.set_xlim(min_year, max_year + 1)
        ax.set_ylim(-1, len(keywords))
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Blue bars for bursts
        for _, row in burst_df.iterrows():
            y = kw_map[row["Keyword"]]
            width = row["End"] - row["Start"]
            # Ensure visible width for 1-year bursts
            width = max(width, 0.8) 
            
            # Color intensity by weight?
            # Let's just use a solid red/blue distinct color
            rect = plt.Rectangle((row["Start"], y - 0.3), width, 0.6, 
                                 facecolor='#d62728', alpha=0.8, edgecolor='none')
            ax.add_patch(rect)
            
            # Add text label for the weight inside the bar if it fits, or to the right
            # ax.text(row["End"] + 0.1, y, f"{row['Weight']:.2f}", va='center', fontsize=8)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords)
        ax.set_xlabel("Year")
        ax.set_title(f"Kleinberg's Keyword Bursts (Top {top_n} Terms)")
        
        # Save
        plt.tight_layout()
        if filename:
             path = os.path.join(self.res_folder, "plots", f"{filename}.png")
             plt.savefig(path, dpi=300, bbox_inches='tight')
             print(f"Saved burst plot to {path}")
             
        plt.show()



    # =========================================================================
    # SLEEPING BEAUTY PLOTTING METHODS
    # =========================================================================

    def plot_sb_overview(
        self, 
        save_path: Optional[str] = None,
        dpi: int = 600
    ) -> plt.Figure:
        """
        Create an overview visualization of all identified Sleeping Beauties.
        
        Args:
            save_path: Path to save the figure (without extension)
            dpi: Resolution for saved figures
        
        Returns:
            matplotlib Figure object
        """
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        return plotbib.plot_sleeping_beauties_overview(
            self.sleeping_beauties, 
            save_path=save_path,
            dpi=dpi
        )

    def plot_sb_trajectory(
        self, 
        paper_index: int = 0,
        show_expected_line: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 600
    ) -> plt.Figure:
        """
        Plot the citation trajectory of a specific Sleeping Beauty.
        
        Args:
            paper_index: Index in the sleeping_beauties DataFrame
            show_expected_line: Whether to show expected linear trajectory
            save_path: Path to save the figure (without extension)
            dpi: Resolution for saved figures
        
        Returns:
            matplotlib Figure object
        """
        paper = self.get_sleeping_beauty_result(paper_index)
        return plotbib.plot_citation_trajectory(
            paper, 
            show_expected_line=show_expected_line,
            save_path=save_path,
            dpi=dpi
        )

    def plot_sb_multi_trajectories(
        self,
        max_papers: int = 10,
        save_path: Optional[str] = None,
        dpi: int = 600
    ) -> plt.Figure:
        """
        Plot citation trajectories for multiple Sleeping Beauties.
        
        Args:
            max_papers: Maximum number of papers to plot
            save_path: Path to save the figure (without extension)
            dpi: Resolution for saved figures
        
        Returns:
            matplotlib Figure object
        """
        if self.sleeping_beauties is None or len(self.sleeping_beauties) == 0:
            raise ValueError("No Sleeping Beauties found. Run extract_sleeping_beauties() first.")
        
        papers = [
            self.get_sleeping_beauty_result(i) 
            for i in range(min(max_papers, len(self.sleeping_beauties)))
        ]
        
        return plotbib.plot_multi_paper_trajectories(
            papers, 
            max_papers=max_papers,
            save_path=save_path,
            dpi=dpi
        )

    def plot_sb_ranking(
        self, 
        top_n: int = 25,
        save_path: Optional[str] = None,
        dpi: int = 600
    ) -> plt.Figure:
        """
        Create a ranking chart of papers by Beauty Coefficient.
        
        Args:
            top_n: Number of top papers to show
            save_path: Path to save the figure (without extension)
            dpi: Resolution for saved figures
        
        Returns:
            matplotlib Figure object
        """
        if self.all_metrics is None:
            raise ValueError("Run compute_sb_metrics() first")
        
        return plotbib.plot_beauty_coefficient_ranking(
            self.all_metrics, 
            top_n=top_n,
            save_path=save_path,
            dpi=dpi
        )

    def plot_sb_timeline(
        self, 
        save_path: Optional[str] = None,
        dpi: int = 600
    ) -> plt.Figure:
        """
        Plot a timeline of Sleeping Beauty publications and awakenings.
        
        Args:
            save_path: Path to save the figure (without extension)
            dpi: Resolution for saved figures
        
        Returns:
            matplotlib Figure object
        """
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        return plotbib.plot_awakening_timeline(
            self.sleeping_beauties, 
            save_path=save_path,
            dpi=dpi
        )

    def generate_all_sb_plots(
        self, 
        output_dir: str,
        dpi: int = 600
    ) -> None:
        """
        Generate and save all Sleeping Beauty visualization plots.
        
        Args:
            output_dir: Directory to save all plots
            dpi: Resolution for saved figures
        """
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        print("\nGenerating all Sleeping Beauty visualizations...")
        
        # Overview plot
        self.plot_sb_overview(save_path=f"{output_dir}/sb_overview", dpi=dpi)
        plt.close()
        
        # Ranking plot
        self.plot_sb_ranking(save_path=f"{output_dir}/sb_ranking", dpi=dpi)
        plt.close()
        
        # Timeline plot
        self.plot_sb_timeline(save_path=f"{output_dir}/sb_timeline", dpi=dpi)
        plt.close()
        
        # Top paper trajectory
        if len(self.sleeping_beauties) > 0:
            self.plot_sb_trajectory(
                paper_index=0, 
                save_path=f"{output_dir}/sb_top_trajectory",
                dpi=dpi
            )
            plt.close()
        
        # Multi-trajectory comparison
        if len(self.sleeping_beauties) >= 3:
            self.plot_sb_multi_trajectories(
                save_path=f"{output_dir}/sb_multi_trajectory",
                dpi=dpi
            )
            plt.close()
        
        print("All Sleeping Beauty plots generated successfully!")

    # =========================================================================
    # SLEEPING BEAUTY EXPORT METHODS
    # =========================================================================

    def save_sb_results(self, output_dir: str) -> None:
        """
        Save Sleeping Beauty analysis results to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        print("\nSaving Sleeping Beauty results...")
        
        # Save Sleeping Beauties
        sb_export = self.sleeping_beauties.drop(columns=["citation_history"], errors="ignore")
        sb_export.to_csv(f"{output_dir}/sleeping_beauties.csv", index=False)
        print(f"   - Saved: sleeping_beauties.csv")
        
        # Save all metrics
        if self.all_metrics is not None:
            metrics_export = self.all_metrics.drop(columns=["citation_history"], errors="ignore")
            metrics_export.to_csv(f"{output_dir}/all_papers_sb_metrics.csv", index=False)
            print(f"   - Saved: all_papers_sb_metrics.csv")
        
        # Save Storytellers
        if self.storytellers is not None and len(self.storytellers) > 0:
            self.storytellers.to_csv(f"{output_dir}/storytellers.csv", index=False)
            print(f"   - Saved: storytellers.csv")

    def print_sb_summary(self) -> None:
        """Print a formatted summary of the Sleeping Beauty analysis results."""
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        print("\n" + "=" * 60)
        print("SLEEPING BEAUTY ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal papers analyzed: {len(self.df)}")
        if self.all_metrics is not None:
            print(f"Papers with metrics: {len(self.all_metrics)}")
        print(f"Sleeping Beauties found: {len(self.sleeping_beauties)}")
        if self.storytellers is not None:
            print(f"Storytellers identified: {len(self.storytellers)}")
        
        if len(self.sleeping_beauties) > 0:
            print(f"\nSleeping Beauty Statistics:")
            print(f"   Average Beauty Coefficient: {self.sleeping_beauties['beauty_coefficient'].mean():.1f}")
            print(f"   Maximum Beauty Coefficient: {self.sleeping_beauties['beauty_coefficient'].max():.1f}")
            print(f"   Average Sleep Duration: {self.sleeping_beauties['sleep_duration'].mean():.1f} years")
            print(f"   Maximum Sleep Duration: {self.sleeping_beauties['sleep_duration'].max():.0f} years")
        
            print(f"\nTop 5 Sleeping Beauties:")
            print("-" * 50)
            for i, (idx, row) in enumerate(self.sleeping_beauties.head(5).iterrows()):
                print(f"\n{i+1}. {row['title'][:65]}...")
                print(f"   Year: {row['publication_year']} | B: {row['beauty_coefficient']:.1f} | "
                      f"Sleep: {row['sleep_duration']:.0f} yrs | Citations: {row['total_citations']}")





from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is only imported for type checkers / IDEs, not at runtime.
    from matplotlib.axes import Axes    
    
class BiblioGroupPlot(BiblioGroup):
    
    def plot_group_overlapping(self, plot_types=["venn", "upset", "heatmap","dendrogram"],
                               title=None, filename="overapping", show=True,
                               include_totals_venn=True,
                               alpha_venn=0.5,
                               methods_heatmap=["jaccard"],
                               color_ticks_heatmap=False,
                               save_csv_heatmap=True,
                               threshold_chord=0.0,
                               method_dendrogram="average",
                               metric_dendrogram="euclidean",
                               # Network parameters
                               method_network="jaccard",
                               threshold_network=0.1,
                               layout_network="spring",
                               **kwargs):

        """Visualise the overlap between document groups.

        This method uses ``group_matrix`` to compute overlaps between groups
        and can display them as Venn diagrams, UpSet plots, heatmaps,
        chord diagrams, dendrograms, or network graphs depending on the chosen ``kind``.

        Parameters
        ----------
        plot_types : list of str, default ["venn", "upset", "heatmap", "dendrogram"]
            Types of visualisation to create. Options: "venn", "upset", "heatmap", 
            "chord", "dendrogram", "network".
        min_group_size : int, default 1
            Minimum number of documents for a group to be included.
        filename : str, default "group_overlap"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        method_network : str, default "jaccard"
            Similarity method for network: 'jaccard', 'count', 'dice', 'overlap'.
        threshold_network : float, default 0.1
            Minimum similarity to show an edge in network.
        layout_network : str, default "spring"
            Network layout: 'spring', 'circular', 'kamada_kawai', 'shell'.
        **kwargs :
            Additional keyword arguments forwarded to the underlying plotting
            routines in :mod:`plotbib`.
        """
        filename = os.path.join(self.res_folder, "plots", filename + "_")

        for plot_type in plot_types:
                    
            if plot_type == "venn":
                plotbib.plot_group_venn(self.group_matrix, title=title, filename=filename+"venn", dpi=self.dpi, include_totals=include_totals_venn, show=show, save_results=True, group_color=self.group_colors, alpha=alpha_venn, **kwargs)
            if plot_type == "upset":
                plotbib.plot_group_upset(self.group_matrix, title=title, filename=filename+"upset", dpi=self.dpi, show=show, save_results=True, group_color=self.group_colors, **kwargs)
            if plot_type == "heatmap":
                plotbib.plot_group_heatmap(self.group_matrix, methods=methods_heatmap, title=title, filename=filename+"heatmap", dpi=self.dpi, group_color=self.group_colors, color_ticks=color_ticks_heatmap, show=show, save_results=True, save_csv=save_csv_heatmap, **kwargs)
            if plot_type == "chord": # to be fixed
                plotbib.plot_group_chord(self.group_matrix, threshold=threshold_chord, group_color=self.group_colors, title=title, filename=filename+"chord", dpi=self.dpi, show=show)
            if plot_type == "dendrogram":
                plotbib.plot_group_dendrogram(self.group_matrix, method=method_dendrogram, metric=metric_dendrogram, title=title, filename=filename+"dendrogram", dpi=self.dpi, show=show)
            if plot_type == "network":
                plotbib.plot_group_intersection_network(self.group_matrix, method=method_network, threshold=threshold_network, group_color=self.group_colors, title=title, filename=filename+"network", dpi=self.dpi, show=show, save_results=True, layout=layout_network, **kwargs)

    def plot_group_intersection_network(self, method="jaccard", threshold=0.1, 
                                         layout="spring", title=None, 
                                         filename="intersection_network", show=True,
                                         **kwargs):
        """
        Plot a network visualization of group intersections/similarities.
        
        Nodes represent groups (sized by number of documents).
        Edges represent overlap/similarity between groups.
        
        Parameters
        ----------
        method : str, default "jaccard"
            Similarity method: 'jaccard', 'count', 'dice', 'overlap'.
        threshold : float, default 0.1
            Minimum similarity to show an edge.
        layout : str, default "spring"
            Network layout: 'spring', 'circular', 'kamada_kawai', 'shell'.
        title : str, optional
            Title for the plot.
        filename : str, default "intersection_network"
            Base filename for saving.
        show : bool, default True
            Whether to display the plot.
        **kwargs
            Additional arguments passed to plot_group_intersection_network.
        
        Returns
        -------
        tuple
            (fig, ax, G) - figure, axes, and networkx graph object.
        """
        filename_path = os.path.join(self.res_folder, "plots", filename)
        return plotbib.plot_group_intersection_network(
            self.group_matrix,
            method=method,
            threshold=threshold,
            group_color=self.group_colors,
            title=title,
            filename=filename_path,
            dpi=self.dpi,
            show=show,
            save_results=True,
            layout=layout,
            **kwargs
        )


    def plot_top_items(self, items=["sources", "countries"], top_n=5, value_column_pattern="Number of documents",
                       title=None, filename="top", show_values=True,
                       reverse_order=False, show=True):
        
        """Plot top items per group for one or more mapping entries.

        For each entry in ``items`` the function extracts group-level
        statistics from the internal mapping and produces a bar chart or
        similar representation of the top ``top_n`` items for each group.

        Parameters
        ----------
        items : sequence of str, default ("sources", "countries")
            Mapping keys for which group-level statistics are available.
        top_n : int, default 5
            Number of top items to display per group.
        value_column_pattern : str, default "Number of documents"
            Pattern used to select the metric column (for example
            "Number of documents" or "Total citations").
        filename_base : str, default "top items"
            Base filename (without extension) for saving the plots into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to the plotting helper
            functions in :mod:`plotbib`.
        """
        filename = os.path.join(self.res_folder, "plots", filename + "_")
        
        for item in items:
            d = self.mapping[item]
            if not hasattr(self, d["group counts df"]):
                getattr(self, d["counter groups"])()
            df = getattr(self, d["group counts df"])
            
            plotbib.plot_top_items_by_group(df, top_n=top_n,
                                 value_column_pattern=value_column_pattern,
                                 title=title,
                                 filename=filename+item,
                                 dpi=self.dpi,
                                 group_color=self.group_colors,
                                 show_values=show_values,
                                 reverse_order=reverse_order,
                                 show=show)
            
    def plot_c_vars_across_groups(self, numerical_cols=["Year", "Cited by", "Sentiment Score", "Interdisciplinarity"],
                                  plot_types=["histogram", "violin plot", "boxplot"], file_name="group comparison",
                                  group_colors=True, bins=30, alpha=0.7, show_grid=False, **kwargs):
        """Plot numerical variables across groups.

        For each numerical column the method creates a distribution plot
        (box, violin or similar) faceted by group. The data are taken from
        the instance data frame.

        Parameters
        ----------
        numerical_cols : sequence of str, optional
            Numerical columns to plot. If None, a sensible default set is used.
        group_col : str, default "Group"
            Column containing group labels.
        filename_base : str, default "c vars across groups"
            Base filename (without extension) for saving the plots into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_numeric_by_group`.
        """
        # Filter out kwargs that are not accepted by plot functions
        # These might be passed by mistake from other contexts
        ignored_kwargs = {"var", "top_n", "n", "items"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignored_kwargs}
        
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]
        group_colors = self.group_colors if group_colors else  {}
        
        if file_name is not None:
            file_name = os.path.join(self.res_folder, "plots", file_name)
            save = True
        else:
            save=False
        
        for plot_type in plot_types:
            if plot_type == "histogram":
                plotbib.plot_group_distributions_aligned(self.df, numerical_cols, self.group_matrix,
                                                         bins=bins, alpha=alpha, 
                                                     save=save, filename_prefix=file_name+"_histogram", dpi=self.dpi, 
                                                     show_grid=show_grid, group_colors=group_colors)
            if plot_type in ["boxplot", "box"]:
                for value_column in numerical_cols:
                    plotbib.plot_boxplot(self.df, value_column, group_matrix=self.group_matrix, filename_base=file_name+"_boxplot", dpi=self.dpi, group_colors=group_colors, **filtered_kwargs)
            if plot_type in ["violin plot", "violin"]:
                for value_column in numerical_cols:
                    plotbib.plot_violinplot(self.df, value_column, group_matrix=self.group_matrix, filename_base=file_name+"_violin", dpi=self.dpi, group_colors=group_colors, **filtered_kwargs)
    

    def plot_stacked_production_by_group(self, filename_base="production by group",
                                         figsize=(10,6), cut_year=None, year_span=None,
                                         font_size=12, xlabel="Year", ylabel="Number of documents",
                                         citation_mode="group",
                                         citation_label="Cumulative Citations", legend_title="Group",
                                         grid=False,
                                         **kwargs):
        
        """Plot stacked scientific production by group over time.

        This method relies on a pre-computed production data frame created by
        :meth:`get_scientific_production`. The data are then visualised with
        :func:`plotbib.plot_stacked_production_by_group`.

        Parameters
        ----------
        filename_base : str, default "production by group"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        grid : bool, default False
            Whether to show grid lines.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_stacked_production_by_group`.
        """
        filename_base = os.path.join(self.res_folder, "plots", filename_base)
        if not hasattr(self, "production_df"):
            self.get_scientific_production(**kwargs)
        
        
        plotbib.plot_stacked_production_by_group(self.production_df,
           group_colors=self.group_colors,
           filename_base=filename_base,
           figsize=figsize,
           cut_year=cut_year,
           year_span=year_span,
           citation_mode=citation_mode,
           font_size=font_size,
           xlabel=xlabel,
           ylabel=ylabel,
           citation_label=citation_label,
           legend_title=legend_title,
           grid=grid)
    
 
    def plot_group_metric_heatmap(
        self,
        items: str,
        metric: str = "Number of documents",
        top_k: int = 20,
        groups: list[str] | None = None,
        title: str | None = None,
        filename: str | None = None,
        dpi: int = 600,
        **kwargs,
    ):
        """
        Plot a heatmap of a metric by item and group.
    
        This method resolves the appropriate group statistics DataFrame from
        ``self.mapping[items]["stats df groups"]``, reshapes it into a long
        format with columns "entity" and "group", optionally filters the
        data, and then delegates to :func:`plotbib.plot_group_metric_heatmap`.
    
        Parameters
        ----------
        items : str
            Mapping key describing the item type (for example "authors" or
            "sources").
        metric : str, default "Number of documents"
            Name of the metric column in the group statistics DataFrame.
        top_k : int, default 20
            Maximum number of entities to keep. If ``top_k <= 0``, all
            entities are kept.
        groups : list of str or None, optional
            If provided, restrict the heatmap to these groups only. Values
            must match the labels in the group column of the statistics
            DataFrame.
        title : str or None, optional
            Custom title for the plot.
        filename : str or None, optional
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``. If ``None``, a name is derived from
            ``items`` and ``metric``.
        dpi : int, default 600
            Resolution for saved figures.
        **kwargs :
            Additional keyword arguments forwarded to
            :func:`plotbib.plot_group_metric_heatmap`.
    
        Returns
        -------
        matplotlib.figure.Figure
            The figure returned by :func:`plotbib.plot_group_metric_heatmap`.
        """
        import os
        import pandas as pd
    
        mapping_entry = self.mapping.get(items, {})
        if not mapping_entry:
            raise KeyError(f"No mapping entry found for items={items!r} in self.mapping.")
    
        key_or_df = mapping_entry.get("stats df groups")
        if key_or_df is None:
            raise KeyError(
                f"'stats df groups' not found for items={items!r} in self.mapping."
            )
    
        stats_df = getattr(self, key_or_df) if isinstance(key_or_df, str) else key_or_df
        if not isinstance(stats_df, pd.DataFrame):
            raise TypeError(
                f"Resolved object for items={items!r} is not a DataFrame "
                f"(got {type(stats_df).__name__})."
            )
    
        df = stats_df.copy()
    
        item_col = mapping_entry.get("label", "Item")
        group_col = mapping_entry.get("group column", "Group")
    
        def _ensure_column(data: pd.DataFrame, name: str) -> pd.DataFrame:
            """Ensure that *name* is a column (promote index level if needed)."""
            if name in data.columns:
                return data
    
            if isinstance(data.index, pd.MultiIndex):
                if name in data.index.names:
                    return data.reset_index(level=name)
            else:
                if data.index.name == name:
                    return data.reset_index()
    
            raise KeyError(
                f"Column or index level {name!r} not found in stats_df "
                f"for items={items!r}."
            )
    
        # Make sure group and item are columns
        df = _ensure_column(df, group_col)
        df = _ensure_column(df, item_col)
    
        if metric not in df.columns:
            raise KeyError(
                f"Metric column {metric!r} not found in stats_df for items={items!r}."
            )
    
        # Optional filtering by groups
        if groups is not None:
            df = df[df[group_col].isin(groups)]
    
        df = df.dropna(subset=[metric])
        if df.empty:
            raise ValueError(
                f"No data available to plot for items={items!r} after filtering."
            )
    
        # Pre-select top_k entities by total metric across groups
        if top_k and top_k > 0:
            totals = (
                df.groupby(item_col, dropna=False)[metric]
                .sum()
                .sort_values(ascending=False)
            )
            keep_entities = totals.head(top_k).index
            df = df[df[item_col].isin(keep_entities)]
    
        if df.empty:
            raise ValueError(
                f"No data left after selecting top_k={top_k} items "
                f"for items={items!r}."
            )
    
        # Rename to the standard names expected by the helper
        df_for_plot = df.rename(columns={item_col: "entity", group_col: "group"})
    
        # Build filename base
        if filename is None:
            safe_items = str(items).strip().replace(" ", "_").lower()
            safe_metric = str(metric).strip().replace(" ", "_").lower()
            filename = f"group_heatmap_{safe_items}_{safe_metric}"
    
        filename_base = None
        if getattr(self, "res_folder", None):
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            filename_base = os.path.join(plots_dir, filename)
    
        return plotbib.plot_group_metric_heatmap(
            stats_df=df_for_plot,
            metric=metric,
            top_k=top_k,
            title=title,
            filename_base=filename_base,
            dpi=dpi,
            **kwargs,
        )

    
    
    def plot_group_metric_bubblemap(
        self,
        items,
        metric: str = "H-index",      # color metric; size = Number of documents
        top_k: int = 25,
        title: str | None = None,
        filename: str | None = None,
        dpi: int = 600,
        **kwargs,                      # e.g., gap=0.06, xpad_extra=0.2, ypad_extra=0.2
    ):
        """Plot a bubble map of group-level metrics for selected items.

        The method uses group statistics from ``mapping["stats df groups"]``
        and visualises them as a scatter/bubble plot where bubble size and/or
        colour encode the chosen metric.

        Parameters
        ----------
        items : str, default "authors"
            Mapping key describing the item type.
        metric : str, default "Number of documents"
            Metric to display.
        filename_base : str, default "group metric bubblemap"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to the plotting helper in
            :mod:`plotbib`.
        """
        key_or_df = self.mapping.get(items, {}).get("stats df groups", None)
        if key_or_df is None:
            raise KeyError(f"'stats df groups' not found for items={items!r} in self.mapping.")
        stats_df = getattr(self, key_or_df) if isinstance(key_or_df, str) else key_or_df
        if not isinstance(stats_df, pd.DataFrame):
            raise TypeError(f"Resolved object for items={items!r} is not a DataFrame (got {type(stats_df).__name__}).")
    
        if filename is None:
            safe_metric = str(metric).lower().replace(" ", "_")
            filename = f"bubblemap_size_docs_color_{safe_metric}"
        filename_base = None
        if getattr(self, "res_folder", None):
            plots_dir = os.path.join(self.res_folder, "plots"); os.makedirs(plots_dir, exist_ok=True)
            filename_base = os.path.join(plots_dir, filename)
    
        return plotbib.plot_group_metric_bubblemap(
            stats_df=stats_df, metric=metric, top_k=top_k,
            title=title, filename_base=filename_base, dpi=dpi, **kwargs
        )
    
    
    def plot_group_metric_slope(
        self,
        items,
        metric: str = "Number of documents",
        group_a: str | None = None,
        group_b: str | None = None,
        top_k: int = 20,
        title: str | None = None,
        filename: str | None = None,
        dpi: int = 600,
        **kwargs,   # e.g., connector_color="0.75", color_by_change=False, up_color="#2ca02c", down_color="#d62728"
    ):
        # Resolve stats_df from mapping (string attr name or direct df)
        """Plot a slope chart comparing group-level metrics between two points.

        Typically used to show how the metric for each item changes between two
        groups or two time slices.

        Parameters
        ----------
        items : str, default "authors"
            Mapping key describing the item type.
        metric : str, default "Number of documents"
            Metric to compare.
        filename_base : str, default "group metric slope"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to the plotting helper in
            :mod:`plotbib`.
        """
        key_or_df = self.mapping.get(items, {}).get("stats df groups", None)
        if key_or_df is None:
            raise KeyError(f"'stats df groups' not found for items={items!r} in self.mapping.")
        stats_df = getattr(self, key_or_df) if isinstance(key_or_df, str) else key_or_df
        if not isinstance(stats_df, pd.DataFrame):
            raise TypeError(f"Resolved object for items={items!r} is not a DataFrame (got {type(stats_df).__name__}).")
    
        # --- infer most frequent groups if needed ---
        if isinstance(stats_df.index, pd.MultiIndex) and "group" not in stats_df.columns:
            gname = stats_df.index.names[0] or "group"
            groups_series = pd.Series(stats_df.index.get_level_values(gname))
        else:
            grp_col = "group"
            if "group" not in stats_df.columns:
                for c in stats_df.columns:
                    if str(c).lower() == "group":
                        grp_col = c
                        break
            if grp_col not in stats_df.columns:
                raise ValueError("Cannot determine group labels (no MultiIndex level or 'group' column).")
            groups_series = stats_df[grp_col]
    
        counts = groups_series.value_counts(dropna=True)
        available = counts.index.tolist()
    
        if group_a is None and group_b is None:
            if len(available) < 2:
                raise ValueError("Need at least two groups to auto-select for slope chart.")
            group_a, group_b = available[:2]
        elif group_a is None:
            group_a = next((g for g in available if g != group_b), None)
        elif group_b is None:
            group_b = next((g for g in available if g != group_a), None)
        if group_a is None or group_b is None or group_a == group_b:
            raise ValueError("Could not infer two distinct groups for the slope chart.")
    
        # --- filename base ---
        if filename is None:
            safe_metric = str(metric).lower().replace(" ", "_")
            safe_a = str(group_a).lower().replace(" ", "_")
            safe_b = str(group_b).lower().replace(" ", "_")
            filename = f"slope_{safe_metric}_{safe_a}_to_{safe_b}"
        filename_base = None
        if getattr(self, "res_folder", None):
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            filename_base = os.path.join(plots_dir, filename)
    
        # --- delegate to plotbib ---
        return plotbib.plot_group_metric_slope(
            stats_df=stats_df,
            metric=metric,
            group_a=group_a,
            group_b=group_b,
            top_k=top_k,
            title=title,
            filename_base=filename_base,
            dpi=dpi,
            **kwargs,
        )

   
    
    def plot_group_metric_bump(
        self,
        items,
        metric: str = "Number of documents",
        entities: list[str] | None = None,
        top_k_auto: int = 12,
        title: str | None = None,
        filename: str | None = None,
        dpi: int = 600,
        **kwargs,   # e.g., groups_order=[...], show_end_labels=True, label_side="right", min_label_gap=0.28
    ):
        """Plot a bump chart of item ranks across groups.

        Ranks for the selected metric are computed within each group and
        displayed as a bump chart to highlight how item importance changes
        between groups.

        Parameters
        ----------
        items : str, default "authors"
            Mapping key describing the item type.
        metric : str, default "Number of documents"
            Metric used for ranking.
        filename_base : str, default "group metric bump"
            Base filename (without extension) for saving the plot into
            ``<res_folder>/plots``.
        **kwargs :
            Additional keyword arguments forwarded to the plotting helper in
            :mod:`plotbib`.
        """
        key_or_df = self.mapping.get(items, {}).get("stats df groups", None)
        if key_or_df is None:
            raise KeyError(f"'stats df groups' not found for items={items!r} in self.mapping.")
        stats_df = getattr(self, key_or_df) if isinstance(key_or_df, str) else key_or_df
        if not isinstance(stats_df, pd.DataFrame):
            raise TypeError(f"Resolved object for items={items!r} is not a DataFrame (got {type(stats_df).__name__}).")
    
        if filename is None:
            safe_metric = str(metric).lower().replace(" ", "_")
            filename = f"bump_{safe_metric}"
        filename_base = None
        if getattr(self, "res_folder", None):
            plots_dir = os.path.join(self.res_folder, "plots"); os.makedirs(plots_dir, exist_ok=True)
            filename_base = os.path.join(plots_dir, filename)
    
        return plotbib.plot_group_metric_bump(
            stats_df=stats_df, metric=metric, entities=entities, top_k_auto=top_k_auto,
            title=title, filename_base=filename_base, dpi=dpi, **kwargs
        )


    
    def plot_average_citations_per_year_by_group_method(
        self,
        filename: str | None = None,
        *,
        year_col: str = "Year",
        cites_col: str = "Cited by",
        doc_prefix: str = "Number of documents ",
        cumcit_prefix: str = "Cumulative Citations ",
        avg_prefix: str = "Average Citations per Document ",
        stats_attr: str = "avg_citations_per_year_by_group_df",
        cut_year: int | None = None,
        plot_overall: bool = True,
        **kwargs,
    ) -> "Axes":
        """
        Compute per-year stats by group using `self.group_matrix`, store them on `self`,
        and plot via `plotbib.plot_average_citations_per_year_by_group` with a continuous time axis.
    
        Produced columns per group:
            "{doc_prefix}<GROUP>"    -> documents published in each year
            "{cumcit_prefix}<GROUP>" -> cumulative sum of citations across years
            "{avg_prefix}<GROUP>"    -> per-year average = sum_citations(year) / docs(year)
    
        The plotting helper accepts `cut_year` (aggregates years < cut_year into "Before <cut_year>")
        and overlays an overall weighted average when `plot_overall=True`. The x-axis is continuous
        from min..max year; missing years are inserted. Line plots forward-fill NaNs to avoid breaks.
    
        Args:
            filename: Base filename (no extension). Saved to "<res_folder>/plots/<filename>" if provided.
            year_col: Year column in `self.df`.
            cites_col: Citations column in `self.df`.
            doc_prefix, cumcit_prefix, avg_prefix: Output column prefixes.
            stats_attr: Attribute name to store the computed wide table on `self`.
            cut_year: Aggregate years below this threshold.
            plot_overall: Whether to draw the overall average line (black).
            **kwargs: Forwarded to the helper (e.g., plot_type, title, etc.). If `save_dpi`
                      is not provided, it defaults to `self.dpi` when available.
    
        Returns:
            Axes: Primary axes from the helper plot.
        """
        if not hasattr(self, "df"):
            raise AttributeError("Expected `self.df` on the instance.")
        if not hasattr(self, "group_matrix"):
            raise AttributeError("Expected `self.group_matrix` on the instance.")
        if year_col not in self.df.columns or cites_col not in self.df.columns:
            raise ValueError(f"`self.df` must contain columns \"{year_col}\" and \"{cites_col}\".")
    
        # Prepare base data
        base_df = self.df[[year_col, cites_col]].copy()
        year_num = pd.to_numeric(base_df[year_col], errors="coerce")
        base_df = base_df.loc[~year_num.isna()].copy()
        base_df[year_col] = year_num.loc[base_df.index].astype(int)
    
        # Align group matrix
        G = self.group_matrix.reindex(self.df.index).fillna(0)
        G = G.astype(float).gt(0).astype(int)
    
        years = np.sort(base_df[year_col].unique())
    
        parts = []
        for grp in G.columns:
            mask = G[grp].astype(bool)
            sub = self.df.loc[mask, [year_col, cites_col]].copy()
    
            if sub.empty:
                part = pd.DataFrame({year_col: years})
                part[f"{doc_prefix}{grp}"] = 0
                part[f"{cumcit_prefix}{grp}"] = 0.0
                part[f"{avg_prefix}{grp}"] = np.nan
                parts.append(part)
                continue
    
            # Per-year docs & citation sums
            doc_counts = sub.groupby(year_col).size().reindex(years, fill_value=0)
            cit_sum_year = sub.groupby(year_col)[cites_col].sum().reindex(years, fill_value=0.0)
            cum_cit = cit_sum_year.cumsum()
    
            with np.errstate(divide="ignore", invalid="ignore"):
                avg_year = cit_sum_year / doc_counts.replace(0, np.nan)
    
            part = pd.DataFrame({year_col: years})
            part[f"{doc_prefix}{grp}"] = doc_counts.values
            part[f"{cumcit_prefix}{grp}"] = cum_cit.values
            part[f"{avg_prefix}{grp}"] = avg_year.values
            parts.append(part)
    
        # Merge per-group parts
        stats_df = parts[0]
        for add in parts[1:]:
            stats_df = stats_df.merge(add, on=year_col, how="outer")
        stats_df = stats_df.sort_values(by=year_col).reset_index(drop=True)
    
        # Store on self
        setattr(self, stats_attr, stats_df)
    
        # Build save path
        filename_base = None
        if filename:
            base_folder = getattr(self, "res_folder", None)
            out_dir = os.path.join(base_folder, "plots") if base_folder else os.path.join(".", "plots")
            os.makedirs(out_dir, exist_ok=True)
            filename_base = os.path.join(out_dir, filename)
    
        group_colors = getattr(self, "group_colors", None)
        if "save_dpi" not in kwargs and hasattr(self, "dpi"):
            kwargs["save_dpi"] = self.dpi
    
        ax = plotbib.plot_average_citations_per_year_by_group(
            stats=stats_df,
            year_col=year_col,
            doc_prefix=doc_prefix,
            cumcit_prefix=cumcit_prefix,
            group_colors=group_colors,
            cut_year=cut_year,
            plot_overall=plot_overall,
            filename_base=filename_base,
            **kwargs,
        )
        return ax
        
        
    def plot_associations_top_n_pairs(
        self,
        items,
        n: int = 20,
        filename_base: str = "top_n_pairs_associations",
        **kwds,
    ):
        """
        Plot associations of top-N item pairs.
    
        Parameters
        ----------
        items : str
            The item type to plot (e.g., "author_keywords", "sources").
        n : int, default 20
            Number of top pairs to display.
        filename_base : str, default "top_n_pairs_associations"
            Base filename for saving. Set to None to skip saving.
        **kwds :
            Additional arguments passed to plotbib.plot_top_n_pairs.
    
        Notes
        -----
        `plotbib.plot_top_n_pairs` expects DataFrame columns "Row" and "Column".
        Your associations store `chi2_sorted_residuals` as a list of tuples
        (row, col, residual, observed, expected) or a DataFrame with lower-case
        names. We now convert/rename before plotting.
    
        Saves under `<res_folder>/relations/` if `filename_base` is not None.
        """
        import os
        import pandas as pd
        from biblium import plotbib
    
        # Normalize items key: replace spaces with underscores
        items_key = items.replace(" ", "_")
    
        # Resolve association object from attribute name (mapping stores strings)
        assoc_attr = None
        if hasattr(self, "mapping") and items_key in self.mapping:
            assoc_attr = self.mapping[items_key].get("associations")
        assoc_attr = assoc_attr or f"{items_key}_associations"
    
        assoc = getattr(self, assoc_attr, None)
        if assoc is None:
            # Provide helpful error message
            available = [a for a in dir(self) if a.endswith("_associations") and getattr(self, a, None) is not None]
            raise AttributeError(
                f"Association object '{assoc_attr}' not found or is None.\n"
                f"Did you call associate_{items_key}() first?\n"
                f"If you did, the association may have returned None because:\n"
                f"  - No items met the min_freq threshold (default=5)\n"
                f"  - The column was empty or missing\n"
                f"Try: associate_{items_key}(min_freq=2) or associate_{items_key}(top_n=50)\n"
                f"Available associations with data: {available}"
            )
    
        pairs = getattr(assoc, "chi2_sorted_residuals", None)
        if pairs is None:
            raise AttributeError(f"'{assoc_attr}' lacks 'chi2_sorted_residuals'.")
    
        # Convert to DataFrame
        if isinstance(pairs, list):
            # list of tuples: (row, col, residual, observed, expected)
            sorted_pairs_df = pd.DataFrame(
                pairs, columns=["row", "col", "residual", "observed", "expected"]
            )
        elif hasattr(pairs, "shape"):
            sorted_pairs_df = pairs.copy()
        else:
            raise TypeError(
                "Unsupported type for chi2_sorted_residuals; expected list of tuples or DataFrame."
            )
    
        # Normalize column names to what plotbib expects
        rename_map = {
            "row": "Row",
            "col": "Column",
            "Row": "Row",
            "Column": "Column",
            # keep metric/size lowercase unless user overrides
            "Residual": "residual",
            "Observed": "observed",
            "Expected": "expected",
        }
        sorted_pairs_df = sorted_pairs_df.rename(columns=rename_map)
    
        # Sanity check for required columns
        for req in ("Row", "Column"):
            if req not in sorted_pairs_df.columns:
                raise KeyError(f"Required column '{req}' missing after normalization.")
    
        # Prepare filename
        if filename_base is not None and getattr(self, "res_folder", None):
            filename_base = os.path.join(self.res_folder, "relations", f"{filename_base}_{items}")
    
        # Axis labels (allow user override)
        x_label = kwds.pop("x_label", "groups")
        y_label = kwds.pop("y_label", items)
    
        # Ensure plotbib has what it needs
        kwds.setdefault("metric_column", "residual")
        kwds.setdefault("size_column", "observed")
    
        # Call the plotter with the normalized DataFrame
        # Use 'n' as 'top_n' parameter
        plotbib.plot_top_n_pairs(
            sorted_pairs_df,
            top_n=n,
            filename_base=filename_base,
            x_label=x_label,
            y_label=y_label,
            **kwds,
        )

    def plot_associations_correspondence_analysis(
        self,
        items,
        filename_base: str = "ca_associations",
        *,
        recompute_if_missing: bool = True,
        n_components: int = 2,
        clean_zeros: bool = True,
        **kwds,
    ):
        """
        Plot correspondence analysis (CA) for item associations.
    
        What this does
        --------------
        - Resolves association/contingency **attribute names** from self.mapping (strings) via getattr(self, ...).
        - If CA coords are missing on the association object and `recompute_if_missing=True`,
          recomputes CA from the contingency with the robust `utilsbib.compute_correspondence_analysis`.
        - Saves under `<res_folder>/relations/<filename_base>_<items>` when `filename_base` is not None.
    
        Parameters
        ----------
        items : str
            Mapping key (e.g., "keywords", "sources").
        filename_base : str, default "ca_associations"
            Base filename (without extension). Set None to skip saving.
        recompute_if_missing : bool, default True
            If the assoc object lacks CA attributes, compute them from the contingency.
        n_components : int, default 2
            CA dimensions to compute when recomputing.
        clean_zeros : bool, default True
            Drop all-zero rows/cols before CA (when recomputing).
        **kwds :
            Forwarded to `plotbib.plot_correspondence_analysis`. User "row_label_name"
            and "col_label_name" override defaults ("groups", items).
        """
        import os
        import pandas as pd
        from biblium import utilsbib
        from biblium import plotbib
    
        # Normalize items key: replace spaces with underscores
        items_key = items.replace(" ", "_")
    
        # Resolve attribute names from mapping; fall back to conventional names
        assoc_attr = cont_attr = None
        if hasattr(self, "mapping") and items_key in self.mapping:
            assoc_attr = self.mapping[items_key].get("associations")
            cont_attr = self.mapping[items_key].get("contingency")
        assoc_attr = assoc_attr or f"{items_key}_associations"
        cont_attr = cont_attr or f"{items_key}_contingency"
    
        assoc = getattr(self, assoc_attr, None)
        cont = getattr(self, cont_attr, None)
    
        if assoc is None:
            raise AttributeError(f"Association object '{assoc_attr}' not found on self.")
        if cont is None or not hasattr(cont, "shape"):
            raise AttributeError(f"Contingency matrix '{cont_attr}' missing on self.")
    
        have_ca = all(hasattr(assoc, a) for a in ("ca_row_coords", "ca_col_coords", "ca_explained_inertia"))
    
        if not have_ca:
            if not recompute_if_missing:
                raise AttributeError(
                    f"'{assoc_attr}' lacks CA attributes and recompute_if_missing=False."
                )
            # Robust numeric coercion for CA
            cont_num = (
                pd.DataFrame(cont)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .astype("float64")
            )
            row_c, col_c, inertia = utilsbib.compute_correspondence_analysis(
                cont_num, n_components=n_components, clean_zeros=clean_zeros
            )
            # Cache back on the assoc object for reuse
            setattr(assoc, "ca_row_coords", row_c)
            setattr(assoc, "ca_col_coords", col_c)
            setattr(assoc, "ca_explained_inertia", inertia)
    
        # Prepare filename in relations/
        if filename_base is not None and getattr(self, "res_folder", None):
            filename_base = os.path.join(self.res_folder, "relations", f"{filename_base}_{items}")
    
        # Labels (allow user overrides)
        row_label_name = kwds.pop("row_label_name", "groups")
        col_label_name = kwds.pop("col_label_name", items)
    
        # Sensible class defaults
        if "dpi" not in kwds and hasattr(self, "dpi"):
            kwds["dpi"] = self.dpi
    
        plotbib.plot_correspondence_analysis(
            assoc.ca_row_coords,
            assoc.ca_col_coords,
            assoc.ca_explained_inertia,
            cont,
            filename_base=filename_base,
            row_label_name=row_label_name,
            col_label_name=col_label_name,
            **kwds,
        )





def get_groups_by_clustering(df, db, preprocess_keywords=False, preprocess_text_vars=True, 
                             text_field="Abstract", method="kmeans", n_clusters=None,
                             k_range=range(2,11), coupling_fields=None, **kwargs):
    """Create a :class:`BiblioGroupPlot` instance by clustering documents.

    A temporary :class:`BiblioStats` object is created from ``df`` and
    document-level clusters are obtained using the chosen clustering
    method. Optionally, textual fields and keywords can be preprocessed
    before clustering. The resulting grouped data frame is wrapped in a
    :class:`BiblioGroupPlot` object for further analysis and plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        Input bibliographic data.
    db : str
        Name of the source database (for example "Scopus" or "WOS").
    preprocess_keywords : bool, default False
        If True, author and index keywords are preprocessed before
        clustering.
    preprocess_text_vars : bool, default True
        If True, long textual fields (such as abstracts) are preprocessed
        before clustering.
    **kwargs :
        Additional keyword arguments forwarded to
        :meth:`BiblioStats.cluster_documents`.

    Returns
    -------
    BiblioGroupPlot
        Wrapper object containing the clustered data frame and convenience
        plotting methods.
    """
    ba = BiblioStats(df=df, db=db)
    if preprocess_keywords:
        ba.process_keywords(**kwargs)
    if preprocess_text_vars:
        ba.process_text_vars(**kwargs)
    text_field = [f for f in ba.df.columns if f in ["Processed "+text_field, text_field]][0]
    ba.cluster_documents(text_field=text_field, method=method, n_clusters=n_clusters,
                          k_range=k_range, coupling_fields=coupling_fields)
    return BiblioGroupPlot(df=ba.df, db=db, group_desc=ba.new_column)

def get_groups_by_concept(df, db, concept_df, concept_column=None, 
                          preprocess_keywords=False, preprocess_text_vars=True, 
                          **kwargs):
    """Create a :class:`BiblioGroupPlot` instance from external concept labels.

    Documents are first preprocessed (if requested) and then matched
    against an externally supplied concept table. Binary membership
    indicators are added to the data frame and used as group descriptors
    in the resulting :class:`BiblioGroupPlot` object.

    Parameters
    ----------
    df : pandas.DataFrame
        Input bibliographic data.
    db : str
        Name of the source database.
    concept_df : pandas.DataFrame
        Table describing concepts or topics, typically with one column
        containing the concept label and another containing example terms.
    concept_column : str or None, optional
        Column in ``df`` to use for matching concepts. If None, a sensible
        default is chosen (for example "Abstract" or "Title").
    preprocess_keywords : bool, default False
        If True, keyword fields are preprocessed before matching.
    preprocess_text_vars : bool, default True
        If True, long textual fields are preprocessed before matching.
    **kwargs :
        Additional keyword arguments forwarded to the preprocessing and
        matching utilities.

    Returns
    -------
    BiblioGroupPlot
        Wrapper object with group descriptors derived from the concept
        indicators.
    """
    ba = BiblioStats(df=df, db=db)
    if preprocess_keywords:
        ba.process_keywords(**kwargs)
    if preprocess_text_vars:
        ba.process_text_vars(**kwargs)
    if concept_column is None:
        concept_column = utilsbib.first_existing(ba.df, ["Processed Combined Text", "Combined Text", "Processed Abstract", "Processed Title", "Abstract", "Title", "Processed Author Keywords", "Author Keywords"])
        
    ba.df = utilsbib.add_concept_indicators(ba.df, concept_df, text_col=concept_column)
    return BiblioGroupPlot(df=ba.df, db=db, group_desc=ba.df[concept_df.columns])


# =============================================================================
# DISRUPTION INDEX PLOT METHODS
# =============================================================================

class DisruptionPlotMixin:
    """Mixin providing disruption index visualization methods."""
    
    def plot_disruption_distribution(
        self,
        metric: str = 'cd_index',
        figsize: Tuple[int, int] = (10, 6),
        color: str = None,
        show_stats: bool = True,
        filename: str = "disruption_distribution",
        ax = None,
        **kwargs
    ):
        """
        Plot distribution of disruption index.
        
        Parameters
        ----------
        metric : str
            Which metric to plot ('cd_index' or 'di_index').
        figsize : tuple
            Figure size.
        color : str, optional
            Bar color.
        show_stats : bool
            Show statistics box.
        filename : str
            Base filename for saving.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'disruption_df') or self.disruption_df is None:
            raise ValueError("No disruption data. Run compute_disruption_index() first.")
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Get valid data
        data = self.disruption_df[metric].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No valid disruption data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Histogram
        bins = np.linspace(-1, 1, 41)
        n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, 
                                          edgecolor='white', linewidth=0.5)
        
        # Color bars based on value
        for i, patch in enumerate(patches):
            bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
            if bin_center > 0.25:
                patch.set_facecolor('#27ae60')  # Green for disruptive
            elif bin_center < -0.25:
                patch.set_facecolor('#e74c3c')  # Red for consolidating
            else:
                patch.set_facecolor(color or '#3498db')  # Blue for neutral
        
        # Reference lines
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=0.25, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Disruptive threshold')
        ax.axvline(x=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Consolidating threshold')
        
        # Labels
        ax.set_xlabel(f'{metric.replace("_", " ").title()}', fontsize=11)
        ax.set_ylabel('Number of Documents', fontsize=11)
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=13)
        ax.set_xlim(-1.05, 1.05)
        
        # Statistics box
        if show_stats:
            stats_text = (
                f'n = {len(data):,}\n'
                f'Mean = {data.mean():.3f}\n'
                f'Median = {data.median():.3f}\n'
                f'Std = {data.std():.3f}'
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#27ae60', label='Disruptive (> 0.25)'),
            mpatches.Patch(facecolor=color or '#3498db', label='Neutral'),
            mpatches.Patch(facecolor='#e74c3c', label='Consolidating (< -0.25)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        if hasattr(self, 'res_folder') and self.res_folder:
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, f"{filename}.png"), dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_disruption_by_entity(
        self,
        entity: str = "sources",
        top_n: int = 20,
        metric: str = 'cd_index_mean',
        figsize: Tuple[int, int] = (12, 8),
        show_error: bool = True,
        filename: str = None,
        ax = None,
        **kwargs
    ):
        """
        Plot disruption index by entity (bar chart).
        
        Parameters
        ----------
        entity : str
            Entity type: 'sources', 'authors', 'countries'.
        top_n : int
            Number of entities to show.
        metric : str
            Which metric to plot.
        figsize : tuple
            Figure size.
        show_error : bool
            Show error bars (std).
        filename : str, optional
            Base filename for saving.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'disruption_df') or self.disruption_df is None:
            raise ValueError("No disruption data. Run compute_disruption_index() first.")
        
        import matplotlib.pyplot as plt
        import numpy as np
        from biblium.disruption import aggregate_disruption_by_entity
        
        # Get entity column and dataframe
        id_col = self.mapping.get("unique-id", "unique-id")
        sep = getattr(self, 'default_separator', '; ')
        
        if entity == "sources":
            entity_col = self.mapping.get("Source_title", "Source title")
            title = "Disruption Index by Source"
        elif entity == "authors":
            entity_col = self.mapping.get("Authors", "Authors")
            title = "Disruption Index by Author"
        elif entity == "countries":
            # Find country column
            entity_col = None
            for col in ["Countries", "Country", "All Countries", "Countries of Authors"]:
                if col in self.df.columns:
                    entity_col = col
                    break
            if entity_col is None:
                entity_col = "Countries"
            title = "Disruption Index by Country"
        else:
            entity_col = entity
            title = f"Disruption Index by {entity}"
        
        # Aggregate
        entity_df = aggregate_disruption_by_entity(
            self.disruption_df, self.df, entity_col, id_col,
            sep=sep if entity != "sources" else "|||", min_docs=1
        )
        
        if entity_df is None or len(entity_df) == 0 or metric not in entity_df.columns:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            ax.text(0.5, 0.5, "No disruption data available for this entity type",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Sort and get top N
        df_sorted = entity_df.nlargest(top_n, metric).copy()
        
        # Truncate long names
        df_sorted[entity_col] = df_sorted[entity_col].astype(str).str[:50]
        
        # Colors based on value
        colors = []
        for val in df_sorted[metric]:
            if pd.isna(val):
                colors.append('#bdc3c7')
            elif val > 0.1:
                colors.append('#27ae60')
            elif val < -0.1:
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')
        
        # Bar plot
        y_pos = range(len(df_sorted))
        bars = ax.barh(y_pos, df_sorted[metric], color=colors, alpha=0.8, edgecolor='white')
        
        # Error bars
        std_col = metric.replace('mean', 'std')
        if show_error and std_col in df_sorted.columns:
            ax.errorbar(df_sorted[metric], y_pos, xerr=df_sorted[std_col],
                       fmt='none', ecolor='gray', alpha=0.5, capsize=3)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted[entity_col], fontsize=9)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=13)
        
        # Reference line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add document count as text
        if 'n_documents' in df_sorted.columns:
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                ax.text(ax.get_xlim()[1], i, f'  n={int(row["n_documents"])}',
                       va='center', fontsize=8, color='gray')
        
        ax.invert_yaxis()
        plt.tight_layout()
        
        # Save
        if hasattr(self, 'res_folder') and self.res_folder:
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fname = filename or f"disruption_by_{entity}"
            fig.savefig(os.path.join(plots_dir, f"{fname}.png"), dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_disruption_over_time(
        self,
        metric: str = 'cd_index_mean',
        figsize: Tuple[int, int] = (12, 6),
        show_ci: bool = True,
        filename: str = "disruption_over_time",
        ax = None,
        **kwargs
    ):
        """
        Plot disruption index trend over time.
        
        Parameters
        ----------
        metric : str
            Which metric to plot.
        figsize : tuple
            Figure size.
        show_ci : bool
            Show confidence interval (±1 std).
        filename : str
            Base filename for saving.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'disruption_df') or self.disruption_df is None:
            raise ValueError("No disruption data. Run compute_disruption_index() first.")
        
        import matplotlib.pyplot as plt
        import numpy as np
        from biblium.disruption import aggregate_disruption_by_entity
        
        year_col = self.mapping.get("Year", "Year")
        id_col = self.mapping.get("unique-id", "unique-id")
        
        # Aggregate by year
        year_df = aggregate_disruption_by_entity(
            self.disruption_df, self.df, year_col, id_col,
            sep="|||", min_docs=1
        )
        
        if year_df is None or len(year_df) == 0 or metric not in year_df.columns:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            ax.text(0.5, 0.5, "No disruption data available for time trend",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Sort by year
        df_sorted = year_df.sort_values(year_col).copy()
        
        try:
            years = df_sorted[year_col].astype(int)
        except:
            years = df_sorted[year_col]
        
        values = df_sorted[metric]
        
        # Line plot
        ax.plot(years, values, 'o-', color='#3498db', linewidth=2, markersize=6, label='Mean CD Index')
        
        # Confidence interval
        std_col = metric.replace('mean', 'std')
        if show_ci and std_col in df_sorted.columns:
            ax.fill_between(years, 
                           values - df_sorted[std_col],
                           values + df_sorted[std_col],
                           alpha=0.2, color='#3498db', label='±1 Std')
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Disruptive threshold')
        ax.axhline(y=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Consolidating threshold')
        
        # Labels
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title('Disruption Index Over Time', fontsize=13)
        ax.legend(loc='best', fontsize=9)
        
        # Set y limits
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        
        # Save
        if hasattr(self, 'res_folder') and self.res_folder:
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, f"{filename}.png"), dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_disruption_scatter(
        self,
        x_col: str = "Year",
        size_col: str = None,
        figsize: Tuple[int, int] = (12, 8),
        filename: str = "disruption_scatter",
        ax = None,
        **kwargs
    ):
        """
        Scatter plot of disruption vs. another variable.
        
        Parameters
        ----------
        x_col : str
            Column for x-axis.
        size_col : str, optional
            Column for point size.
        figsize : tuple
            Figure size.
        filename : str
            Base filename for saving.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'disruption_df') or self.disruption_df is None:
            raise ValueError("No disruption data. Run compute_disruption_index() first.")
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from biblium.disruption import add_disruption_to_df
        
        id_col = self.mapping.get("unique-id", "unique-id")
        
        if size_col is None:
            size_col = self.mapping.get("Cited_by", "Cited by")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Merge dataframes
        merged = add_disruption_to_df(self.df, self.disruption_df, id_col)
        
        if x_col not in merged.columns or 'cd_index' not in merged.columns:
            ax.text(0.5, 0.5, f"Column not found: {x_col} or cd_index", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        data = merged[[x_col, 'cd_index', 'interpretation']].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No valid data for scatter plot",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Size
        if size_col in merged.columns:
            sizes = merged.loc[data.index, size_col].fillna(10)
            sizes = np.clip(sizes, 10, 1000)
            sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 200 + 20
        else:
            sizes = 50
        
        # Colors
        color_map = {
            'disruptive': '#27ae60',
            'neutral': '#3498db',
            'consolidating': '#e74c3c',
            'uncited': '#bdc3c7',
        }
        colors = data['interpretation'].map(color_map).fillna('#95a5a6')
        
        # Scatter
        ax.scatter(data[x_col], data['cd_index'], c=colors, s=sizes, alpha=0.6, edgecolor='white')
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.3)
        
        # Labels
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel('CD Index', fontsize=11)
        ax.set_title(f'Disruption Index vs {x_col}', fontsize=13)
        
        # Legend
        legend_elements = [mpatches.Patch(facecolor=c, label=l) 
                           for l, c in color_map.items() if l in data['interpretation'].values]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        ax.set_ylim(-1.1, 1.1)
        plt.tight_layout()
        
        # Save
        if hasattr(self, 'res_folder') and self.res_folder:
            plots_dir = os.path.join(self.res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, f"{filename}.png"), dpi=self.dpi, bbox_inches='tight')
        
        return fig


# Add DisruptionPlotMixin to BiblioPlot
# This is done by modifying the class definition or using monkey patching
# For now, we add the methods directly to BiblioPlot at import time
for method_name in ['plot_disruption_distribution', 'plot_disruption_by_entity', 
                    'plot_disruption_over_time', 'plot_disruption_scatter']:
    setattr(BiblioPlot, method_name, getattr(DisruptionPlotMixin, method_name))