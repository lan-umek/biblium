# -*- coding: utf-8 -*-
"""
Bibliometric laws plotting - Lotka, Bradford, Zipf.

This module contains methods for:
- Lotka's law (author productivity)
- Bradford's law (journal zones)
- Zipf's law (word frequency)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import numpy as np


class LawsPlotsMixin:
    """Mixin class providing bibliometric laws plotting methods."""
    
    def lotka_law(
        self,
        author_col: str = "Authors",
        filename_base: str = "lotka law",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute and plot Lotka's law for author productivity.

        Parameters
        ----------
        author_col : str
            Column containing author information.
        filename_base : str
            Base filename for saving plots.
        **kwargs :
            Additional plot arguments.

        Returns
        -------
        DataFrame
            Lotka distribution data.
        """
        from biblium import plotbib, utilsbib
        import matplotlib.pyplot as plt
        
        # Ensure author counts exist
        if not hasattr(self, "authors_counts_df"):
            self.count_authors()
        
        # Compute Lotka distribution
        lotka_df = utilsbib.compute_lotka_distribution(
            self.authors_counts_df,
            count_col="Number of documents"
        )
        
        # Evaluate fit
        fit = utilsbib.evaluate_lotka_fit(lotka_df)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear plot
        axes[0].bar(lotka_df["Documents"], lotka_df["Authors"])
        axes[0].set_xlabel("Number of Documents")
        axes[0].set_ylabel("Number of Authors")
        axes[0].set_title("Author Productivity Distribution")
        
        # Log-log plot with fit
        axes[1].scatter(
            np.log10(lotka_df["Documents"]),
            np.log10(lotka_df["Authors"]),
            label="Observed"
        )
        
        # Add fit line
        x = np.log10(lotka_df["Documents"].values)
        y_pred = -fit["exponent"] * x + np.log10(lotka_df["Authors"].iloc[0])
        axes[1].plot(x, y_pred, "r--", label=f"Fit (β={fit['exponent']:.2f})")
        
        axes[1].set_xlabel("Log10(Documents)")
        axes[1].set_ylabel("Log10(Authors)")
        axes[1].set_title(f"Lotka's Law (R²={fit['r_squared']:.3f})")
        axes[1].legend()
        
        plt.tight_layout()
        
        if self.res_folder is not None:
            self._save_plot(filename_base)
        
        # Store results
        self.lotka_df = lotka_df
        self.lotka_fit = fit
        
        return lotka_df

    def bradford_law(
        self,
        source_col: str = "Source title",
        filename_base: str = "bradford law",
        n_zones: int = 3,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute and plot Bradford's law for journal zones.

        Parameters
        ----------
        source_col : str
            Column containing source/journal titles.
        filename_base : str
            Base filename for saving plots.
        n_zones : int
            Number of Bradford zones.
        **kwargs :
            Additional plot arguments.

        Returns
        -------
        DataFrame
            Bradford distribution data with zone assignments.
        """
        from biblium import plotbib, utilsbib
        import matplotlib.pyplot as plt
        
        # Ensure source counts exist
        if not hasattr(self, "sources_counts_df"):
            self.count_sources()
        
        # Compute Bradford distribution
        bradford_df = utilsbib.compute_bradford_distribution(
            self.sources_counts_df,
            count_col="Number of documents"
        )
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cumulative articles vs sources (log scale)
        axes[0].plot(
            range(1, len(bradford_df) + 1),
            bradford_df["Cumulative Docs"],
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Number of Sources (log)")
        axes[0].set_ylabel("Cumulative Documents")
        axes[0].set_title("Bradford's Law - Cumulative Distribution")
        
        # Zone distribution
        zone_counts = bradford_df.groupby("Zone").agg({
            "Number of documents": "sum",
            "Zone": "count"
        }).rename(columns={"Zone": "Sources"})
        
        x = range(len(zone_counts))
        width = 0.35
        
        axes[1].bar(
            [i - width/2 for i in x],
            zone_counts["Sources"],
            width,
            label="Sources",
            color="steelblue"
        )
        axes[1].bar(
            [i + width/2 for i in x],
            zone_counts["Number of documents"],
            width,
            label="Documents",
            color="coral"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(zone_counts.index)
        axes[1].set_xlabel("Zone")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Bradford Zones")
        axes[1].legend()
        
        plt.tight_layout()
        
        if self.res_folder is not None:
            self._save_plot(filename_base)
        
        # Store results
        self.bradford_df = bradford_df
        
        return bradford_df

    def zipf_law(
        self,
        text_col: Optional[str] = None,
        filename_base: str = "zipf law",
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute and plot Zipf's law for word frequency.

        Parameters
        ----------
        text_col : str, optional
            Column containing text. If None, uses abstract or keywords.
        filename_base : str
            Base filename for saving plots.
        top_n : int
            Number of top words to analyze.
        **kwargs :
            Additional plot arguments.

        Returns
        -------
        DataFrame
            Zipf distribution data.
        """
        from biblium import plotbib, utilsbib
        import matplotlib.pyplot as plt
        
        # Determine text column
        if text_col is None:
            for col in ["Processed Abstract", "Abstract", "Processed Author Keywords", "Author Keywords"]:
                if col in self.df.columns:
                    text_col = col
                    break
        
        if text_col is None or text_col not in self.df.columns:
            print("No suitable text column found for Zipf analysis")
            return pd.DataFrame()
        
        # Count words using ngrams
        word_counts = utilsbib.count_occurrences(
            self.df,
            text_col,
            count_type="text",
            item_column_name="Word",
            ngram_range=(1, 1),
        )
        
        # Compute Zipf distribution
        zipf_df = utilsbib.compute_zipf_distribution_from_counts(
            word_counts.head(top_n),
            count_col="Number of documents"
        )
        
        # Evaluate fit
        fit = utilsbib.evaluate_zipf_fit(zipf_df)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rank-frequency plot
        axes[0].bar(zipf_df["Rank"].head(30), zipf_df["Number of documents"].head(30))
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Word Frequency Distribution (Top 30)")
        
        # Log-log plot with fit
        axes[1].scatter(
            zipf_df["Log Rank"],
            zipf_df["Log Frequency"],
            alpha=0.6,
            label="Observed"
        )
        
        # Add fit line
        x = zipf_df["Log Rank"].values
        y_pred = -fit["exponent"] * x + zipf_df["Log Frequency"].iloc[0]
        axes[1].plot(x, y_pred, "r--", label=f"Fit (α={fit['exponent']:.2f})")
        
        axes[1].set_xlabel("Log(Rank)")
        axes[1].set_ylabel("Log(Frequency)")
        axes[1].set_title(f"Zipf's Law (R²={fit['r_squared']:.3f})")
        axes[1].legend()
        
        plt.tight_layout()
        
        if self.res_folder is not None:
            self._save_plot(filename_base)
        
        # Store results
        self.zipf_df = zipf_df
        self.zipf_fit = fit
        
        return zipf_df
