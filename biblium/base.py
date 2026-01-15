# -*- coding: utf-8 -*-
"""
Biblium Base Module - Shared initialization and helper methods.

This module provides the common foundation for BiblioAnalysis and BiblioGroupAnalysis,
eliminating code duplication while maintaining backward compatibility.

@author: Lan.Umek
@refactored: Claude (Anthropic)
"""

import os
import pandas as pd
from typing import Callable, Optional, Dict, List, Union


# =============================================================================
# DATABASE SEPARATOR CONFIGURATION
# =============================================================================

SEPARATOR_MAP = {
    # Major databases
    "scopus": "; ",
    "open alex": "|",
    "oa": "|",
    "wos": "; ",
    "pubmed": "; ",
    "dimensions": "; ",
    "lens": "; ",
    # Engineering/CS
    "ieee": "; ",
    "dblp": "; ",
    "arxiv": "; ",
    "semantic scholar": "; ",
    "s2": "; ",
    # Social sciences
    "proquest": "; ",
    "ebsco": "; ",
    "jstor": "; ",
    "psycinfo": "; ",
    "eric": "; ",
    "econlit": "; ",
    # Other sciences
    "mathscinet": "; ",
    "inspec": "; ",
    "georef": "; ",
    "cab": "; ",
    "cinahl": "; ",
    "embase": "; ",
    "cochrane": "; ",
    # Generic formats
    "ris": "; ",
    "bibtex": " and ",
    "bib": " and ",
    "endnote": "; ",
    "zotero": "; ",
    "csv": "; ",
    "crossref": "; ",
    "orcid": "; ",
}


# =============================================================================
# BIBLIO BASE CLASS
# =============================================================================

class BiblioBase:
    """
    Base class providing shared initialization and helper methods for
    BiblioAnalysis and BiblioGroupAnalysis.
    
    This class is not meant to be instantiated directly. It provides:
    - Common initialization logic via _init_common()
    - Helper methods for saving tables and plots (_save_table, _save_plot)
    - Column lookup helper (_get_column)
    - Caching support for expensive computations
    """
    
    # Cache configuration
    _cache_enabled: bool = True
    _instance_cache: Dict = None
    
    def _init_cache(self) -> None:
        """Initialize the instance cache."""
        self._instance_cache = {}
        self._cache_enabled = True
    
    def _get_cached(
        self,
        cache_key: str,
        compute_func: Callable,
        *args,
        **kwargs,
    ):
        """
        Get a cached result or compute and cache it.
        
        Parameters
        ----------
        cache_key : str
            Unique key for this computation.
        compute_func : callable
            Function to compute the result if not cached.
        *args, **kwargs :
            Arguments passed to compute_func.
            
        Returns
        -------
        Any
            The cached or newly computed result.
        """
        if self._instance_cache is None:
            self._init_cache()
        
        if not self._cache_enabled:
            return compute_func(*args, **kwargs)
        
        if cache_key not in self._instance_cache:
            self._instance_cache[cache_key] = compute_func(*args, **kwargs)
        
        return self._instance_cache[cache_key]
    
    def clear_cache(self, prefix: Optional[str] = None) -> int:
        """
        Clear the instance cache.
        
        Parameters
        ----------
        prefix : str, optional
            If provided, only clear keys starting with this prefix.
            
        Returns
        -------
        int
            Number of items cleared.
        """
        if self._instance_cache is None:
            return 0
        
        if prefix is None:
            count = len(self._instance_cache)
            self._instance_cache.clear()
            return count
        
        keys_to_remove = [k for k in self._instance_cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._instance_cache[k]
        return len(keys_to_remove)
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable instance caching."""
        self._cache_enabled = enabled
    
    def get_cache_info(self) -> Dict:
        """
        Get cache information.
        
        Returns
        -------
        dict
            Cache statistics including size and enabled status.
        """
        if self._instance_cache is None:
            return {"size": 0, "enabled": self._cache_enabled, "keys": []}
        return {
            "size": len(self._instance_cache),
            "enabled": self._cache_enabled,
            "keys": list(self._instance_cache.keys()),
        }
    
    def _make_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from prefix and arguments.
        
        Parameters
        ----------
        prefix : str
            Cache key prefix (e.g., "cooc", "network").
        *args, **kwargs :
            Arguments to include in the key.
            
        Returns
        -------
        str
            A unique cache key.
        """
        import hashlib
        
        components = [prefix]
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use shape and a hash of column names
                components.append(f"df:{arg.shape}:{hash(tuple(arg.columns))}")
            elif isinstance(arg, (list, tuple)):
                if all(isinstance(x, str) for x in arg):
                    components.append(str(sorted(arg)))
                else:
                    components.append(str(arg))
            elif arg is None:
                components.append("None")
            else:
                components.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                components.append(f"{k}=df:{v.shape}")
            elif v is None:
                components.append(f"{k}=None")
            else:
                components.append(f"{k}={v}")
        
        combined = "|".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _init_common(
        self,
        f_name: Optional[str] = None,
        db: str = "",
        df: Optional[pd.DataFrame] = None,
        res_folder: Optional[str] = "results",
        output_lang: str = "en",
        preprocess_level: int = 0,
        exclude_list_kw: Optional[List[str]] = None,
        synonyms_kw: Optional[Dict[str, str]] = None,
        lemmatize_kw: bool = False,
        default_keywords: str = "author",
        combine_with_index_keywords: bool = False,
        concept_df: Optional[pd.DataFrame] = None,
        concept_column: Optional[str] = None,
        asjc_map_df: Optional[pd.DataFrame] = None,
        lang_of_docs: str = "en",
        fancy_output: bool = False,
        label_docs: bool = True,
        dpi: int = 600,
        cmap: str = "viridis",
        cmap_disc: str = "tab10",
        default_color: str = "lightblue",
        extra_stopwords: Optional[List[str]] = None,
        specific_stopword_categories: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize common attributes for bibliometric analysis.
        
        This method handles all shared initialization logic between
        BiblioAnalysis and BiblioGroupAnalysis.
        """
        # Initialize cache
        self._init_cache()
        
        # Lazy imports to avoid circular dependencies
        from biblium import readbib, utilsbib
        
        # Store database and language settings
        self.db = db.lower()
        self.output_lang = output_lang
        self.ldf = lambda x: utilsbib.ldf(x, l=output_lang)
        
        # Store plotting configuration
        self.dpi = dpi
        self.cmap = cmap
        self.cmap_disc = cmap_disc
        self.default_color = default_color
        
        # Store enrichment data
        self.concept_df = concept_df
        
        # Load data
        if df is not None:
            self.df = df.copy() if hasattr(df, "copy") else df
        elif f_name is not None:
            self.df = readbib.read_bibfile(f_name, self.db)
        else:
            print(
                self.ldf("No dataset has been provided."),
                self.ldf("The bibliometric analysis cannot be performed."),
            )
            return
        
        self.n = len(self.df)
        
        # Setup database-specific separator
        self.default_separator = SEPARATOR_MAP.get(self.db, "; ")
        
        # Load mappings
        mapping_path = os.path.join(
            os.path.dirname(__file__), "additional files", "mappings.xlsx"
        )
        mapping_df = pd.read_excel(mapping_path, sheet_name="mapping")
        alias_df = pd.read_excel(mapping_path, sheet_name="alias")
        self.mapping = utilsbib.reconstruct_mapping(mapping_df, alias_df)
        
        # Add document labels
        if label_docs:
            self.df["Doc ID"] = [f"Doc {i}" for i in range(1, self.n + 1)]
        
        # Create source abbreviations
        if "Source title" in self.df.columns:
            if "Abbreviated Source Title" not in self.df.columns:
                self.df["Abbreviated Source Title"] = self.df["Source title"].map(
                    utilsbib.abbreviate_words
                )
            self.sources_abb_dict = self.df.set_index("Source title")[
                "Abbreviated Source Title"
            ].to_dict()
        else:
            self.sources_abb_dict = {}
        
        # Setup output folders
        if res_folder is not None:
            self.res_folder = os.path.join(os.getcwd(), res_folder)
            sub_folders = ["plots", "tables", "reports", "networks", "relations"]
            folders = [os.path.join(self.res_folder, s) for s in sub_folders]
            utilsbib.make_folders(folders)
            self.cond_formatting = fancy_output
            self.autofit = fancy_output
        else:
            self.res_folder = None
            self.cond_formatting = False
            self.autofit = False
        
        # Initialize relations dict
        self.relations = {}
        
        # =====================================================================
        # PREPROCESSING LEVELS
        # =====================================================================
        
        # Level 1: Basic enrichment
        if preprocess_level >= 1:
            if "Cited by" in self.df.columns:
                self.df["Percent Rank Cited by"] = (
                    pd.to_numeric(self.df["Cited by"], errors="coerce")
                    .rank(pct=True, method="average")
                    .mul(100)
                    .round(2)
                )
            
            self.df = utilsbib.add_ca_country_df(self.df, self.db)
            self.missings_df, self.missings = utilsbib.check_missing_values(self.df)
            self.df = utilsbib.add_document_labels_abbrev(self.df)
            
            if "Doc ID" in self.df.columns:
                doc_id_index = self.df.set_index("Doc ID")
                if "Document Short Label" in doc_id_index.columns:
                    self.id_short_label_dict = doc_id_index["Document Short Label"].to_dict()
                if "Document Label" in doc_id_index.columns:
                    self.id_label_dict = doc_id_index["Document Label"].to_dict()
            
            if self.db == "scopus" and "Affiliations" in self.df.columns:
                self.df, self.country_collab_matrix = utilsbib.extract_countries_from_affiliations(
                    self.df, aff_column="Affiliations"
                )
        
        # Level 2: Keywords and text processing
        if preprocess_level >= 2:
            self.process_keywords(
                exclude_list=exclude_list_kw,
                synonyms=synonyms_kw,
                lemmatize=lemmatize_kw,
            )
            self.process_text_vars(
                stopwords_file=utilsbib.stopwords_file,
                lang=lang_of_docs,
                extra_stopwords=extra_stopwords,
                exclude_specific_stopwords=specific_stopword_categories,
            )
            self.get_country_collaboration()
            self.df = utilsbib.build_combined_text(
                self.df,
                include_index_keywords=combine_with_index_keywords,
            )
            
            if concept_df is not None:
                if concept_column is None:
                    concept_column = utilsbib.first_existing(
                        self.df,
                        [
                            "Processed Combined Text",
                            "Combined Text",
                            "Processed Abstract",
                            "Processed Title",
                            "Abstract",
                            "Title",
                            "Processed Author Keywords",
                            "Author Keywords",
                        ],
                    )
                self.df = utilsbib.add_concept_indicators(
                    self.df, concept_df, text_col=concept_column
                )
        
        # Level 3: Science mappings
        if preprocess_level >= 3:
            if self.db == "scopus":
                cited_sciences = preprocess_level == 4
                self.add_sciences_scopus(
                    asjc_map_df=asjc_map_df,
                    cited_sciences=cited_sciences,
                )
            
            self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(
                self.df,
                author_col="Author Keywords",
                index_col="Index Keywords",
                sep=self.default_separator,
            )
            self.df = utilsbib.merge_text_columns(
                self.df,
                title_col="Processed Title",
                abstract_col="Processed Abstract",
                author_col="Processed Author Keywords",
                index_col="Processed Index Keywords",
            )
        
        # Level 4: Interdisciplinarity
        if preprocess_level >= 4:
            if hasattr(self, "cited_sciences_df"):
                self.compute_interdisciplinarity_entropy()
        
        # Describe columns
        self.describe_columns()
        
        # Setup default keyword variable
        if default_keywords == "author":
            self.kw_var = (
                "Processed Author Keywords"
                if "Processed Author Keywords" in self.df.columns
                else "Author Keywords"
            )
        elif default_keywords == "index":
            self.kw_var = (
                "Processed Index Keywords"
                if "Processed Index Keywords" in self.df.columns
                else "Index Keywords"
            )
        elif default_keywords in ["both", "author and index"]:
            if "Author and Index Keywords" not in self.df.columns:
                self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(
                    self.df,
                    author_col="Author Keywords",
                    index_col="Index Keywords",
                    sep=self.default_separator,
                )
            self.kw_var = (
                "Processed Author and Index Keywords"
                if "Processed Author and Index Keywords" in self.df.columns
                else "Author and Index Keywords"
            )
        else:
            self.kw_var = None
        
        # Extract author mappings for Scopus
        if ("Author full names" in self.df.columns) and self.db == "scopus":
            self.id_to_author, self.author_to_id = utilsbib.extract_author_mappings(
                self.df, "Author full names"
            )
        
        # Save missing values report
        if self.res_folder is not None and hasattr(self, "missings_df"):
            self._save_table(self.missings_df, "missing values")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _save_table(
        self,
        df: pd.DataFrame,
        name: str,
        subfolder: str = "tables",
    ) -> None:
        """
        Save DataFrame to Excel if res_folder is set.
        
        Parameters
        ----------
        df : DataFrame
            Data to save.
        name : str
            Filename (without extension).
        subfolder : str
            Subfolder within res_folder.
        """
        if self.res_folder is not None and df is not None:
            from biblium import utilsbib
            utilsbib.to_excel_fancy(
                df,
                f_name=os.path.join(self.res_folder, subfolder, f"{name}.xlsx"),
                autofit=getattr(self, "autofit", False),
                conditional_formatting=getattr(self, "cond_formatting", False),
            )
    
    def _save_plot(self, filename_base: str, subfolder: str = "plots") -> None:
        """
        Save current matplotlib figure if res_folder is set.
        
        Parameters
        ----------
        filename_base : str
            Filename without extension.
        subfolder : str
            Subfolder within res_folder.
        """
        if self.res_folder is not None:
            from biblium import utilsbib
            path = os.path.join(self.res_folder, subfolder, filename_base)
            utilsbib.save_plot(path, dpi=self.dpi)
    
    def _get_column(
        self,
        candidates: Union[str, List[str]],
        required: bool = True,
    ) -> Optional[str]:
        """
        Find the first available column from candidates.
        
        Parameters
        ----------
        candidates : str or list
            Single column name or list of candidates in priority order.
        required : bool
            If True, raise error when no column found.
            
        Returns
        -------
        str or None
            Found column name.
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        
        for col in candidates:
            if col in self.df.columns:
                return col
        
        if required:
            raise ValueError(f"None of the columns found: {candidates}")
        return None
    
    def _count_entity(
        self,
        column: Union[str, List[str]],
        item_label: str,
        count_type: str = "list",
        attr_name: Optional[str] = None,
        binary_attr_name: Optional[str] = None,
        save_name: Optional[str] = None,
        sep: Optional[str] = None,
        top_n: int = 0,
        rename_dict: Optional[Dict[str, str]] = None,
        translated_column_name: Optional[str] = None,
        ngram_range: tuple = (1, 1),
        post_process: Optional[callable] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generic entity counting method.
        
        This is the core helper that powers all count_* methods, eliminating
        repetitive code patterns.
        
        Parameters
        ----------
        column : str or list
            Column name(s) to count. If list, uses first available.
        item_label : str
            Label for the item column in output (e.g., "Source", "Keyword").
        count_type : {"single", "list", "text"}
            How to count values in the column.
        attr_name : str, optional
            Attribute name to store result (e.g., "sources_counts_df").
            If None, derives from item_label.
        binary_attr_name : str, optional
            Attribute name for binary indicators (e.g., "binary_sources_df").
        save_name : str, optional
            Filename for saving (without extension). If None, derives from item_label.
        sep : str, optional
            Separator for list values. Uses self.default_separator if None.
        top_n : int
            If > 0, create binary indicators for top N items.
        rename_dict : dict, optional
            Mapping for translated item names.
        translated_column_name : str, optional
            Column name for translated items.
        ngram_range : tuple
            N-gram range for text count_type.
        post_process : callable, optional
            Function to apply to result DataFrame before saving.
        **kwargs
            Additional arguments for binary indicator creation.
            
        Returns
        -------
        DataFrame
            Counts DataFrame with ranks and proportions.
        """
        from biblium import utilsbib
        
        # Resolve column name
        if isinstance(column, list):
            col = self._get_column(column, required=False)
            if col is None:
                # Return empty DataFrame if column not found
                return pd.DataFrame(columns=[item_label, "Number of documents"])
        else:
            col = column
            if col not in self.df.columns:
                return pd.DataFrame(columns=[item_label, "Number of documents"])
        
        # Use default separator if not specified
        if sep is None:
            sep = getattr(self, "default_separator", "; ")
        
        # Build count_occurrences kwargs
        count_kwargs = {
            "count_type": count_type,
            "item_column_name": item_label,
            "sep": sep,
        }
        
        if count_type == "text":
            count_kwargs["ngram_range"] = ngram_range
        
        if rename_dict is not None:
            count_kwargs["rename_dict"] = rename_dict
            if translated_column_name:
                count_kwargs["translated_column_name"] = translated_column_name
        
        # Perform counting
        result_df = utilsbib.count_occurrences(self.df, col, **count_kwargs)
        
        # Apply post-processing if provided
        if post_process is not None:
            result_df = post_process(result_df)
        
        # Store as attribute
        if attr_name is None:
            attr_name = f"{item_label.lower().replace(' ', '_').replace('-', '_')}_counts_df"
        setattr(self, attr_name, result_df)
        
        # Create binary indicators if requested (with caching)
        if top_n > 0 and not result_df.empty:
            top_items = result_df[item_label].head(top_n).tolist()
            value_type = "string" if count_type == "single" else count_type
            
            # Check cache for binary indicators
            cache_key = None
            indicators_dict = None
            if hasattr(self, '_instance_cache') and self._instance_cache is not None and self._cache_enabled:
                cache_key = self._make_cache_key(
                    "binary", col, tuple(top_items[:20]) if len(top_items) > 20 else tuple(top_items),
                    value_type=value_type
                )
                indicators_dict = self._instance_cache.get(cache_key)
            
            if indicators_dict is None:
                _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                    df=self.df,
                    col=col,
                    items_of_interest=top_items,
                    value_type=value_type,
                    separator=sep,
                    **kwargs,
                )
                if cache_key is not None and self._instance_cache is not None:
                    self._instance_cache[cache_key] = indicators_dict
            
            if binary_attr_name is None:
                binary_attr_name = f"binary_{item_label.lower().replace(' ', '_').replace('-', '_')}_df"
            setattr(self, binary_attr_name, indicators_dict.get("binary"))
        
        # Save to Excel
        if save_name is None:
            save_name = f"{item_label.lower()} counts"
        self._save_table(result_df, save_name)
        
        return result_df
    
    def _get_entity_stats(
        self,
        column: Union[str, List[str]],
        item_label: str,
        count_method_name: str,
        attr_name: str,
        indicators_attr_name: str,
        save_name: str,
        value_type: str = "list",
        top_n: int = 100,
        post_process: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generic helper for get_*_stats methods.
        
        This consolidates the common pattern used by get_sources_stats,
        get_author_keywords_stats, get_authors_stats, etc.
        
        Parameters
        ----------
        column : str or list of str
            Column name(s) to analyze (first existing will be used).
        item_label : str
            Label for items in the result (e.g., "Source", "Keyword").
        count_method_name : str
            Name of the count method to call (e.g., "count_sources").
        attr_name : str
            Attribute name to store results (e.g., "sources_stats_df").
        indicators_attr_name : str
            Attribute name for indicators (e.g., "sources_indicators").
        save_name : str
            Name for saving to Excel.
        value_type : str
            "string" for single values, "list" for delimited lists.
        top_n : int
            Number of top items to analyze (default 100).
        post_process : callable, optional
            Function to apply to result DataFrame.
        **kwargs :
            Additional arguments for get_entity_stats.
        
        Returns
        -------
        DataFrame
            Statistics for the entity.
        """
        from biblium import utilsbib
        
        # Resolve column
        if isinstance(column, str):
            col = column if column in self.df.columns else None
        else:
            col = self._get_column(column, required=False)
        
        if col is None:
            return pd.DataFrame()
        
        # Get count method
        count_method = getattr(self, count_method_name, None)
        if count_method is None:
            return pd.DataFrame()
        
        # Call get_entity_stats
        stats_df, indicators = utilsbib.get_entity_stats(
            self.df,
            col,
            item_label,
            count_method=count_method,
            value_type=value_type,
            sep=getattr(self, "default_separator", "; "),
            top_n=top_n,
            **kwargs,
        )
        
        # Apply post-processing if provided
        if post_process is not None and stats_df is not None:
            stats_df = post_process(stats_df)
        
        # Store results
        setattr(self, attr_name, stats_df)
        setattr(self, indicators_attr_name, indicators)
        
        # Save
        self._save_table(stats_df, save_name)
        
        return stats_df
    
    # =========================================================================
    # PLACEHOLDER METHODS (implemented in subclasses)
    # =========================================================================
    
    def describe_columns(self) -> None:
        """Describe DataFrame columns. Override in subclass."""
        pass
    
    def process_keywords(self, **kwargs) -> None:
        """Process keywords. Override in subclass."""
        pass
    
    def process_text_vars(self, **kwargs) -> None:
        """Process text variables. Override in subclass."""
        pass
    
    def get_country_collaboration(self, **kwargs) -> None:
        """Get country collaboration. Override in subclass."""
        pass
    
    def add_sciences_scopus(self, **kwargs) -> None:
        """Add Scopus sciences. Override in subclass."""
        pass
    
    def compute_interdisciplinarity_entropy(self, **kwargs) -> None:
        """Compute interdisciplinarity. Override in subclass."""
        pass
