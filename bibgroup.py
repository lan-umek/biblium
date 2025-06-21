# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 23:51:43 2025

@author: Lan
"""

import utilsbib
from bibstats import BiblioStats
import pandas as pd
from functools import reduce

def initialize_biblio_common(*args, **kwargs):
    temp = BiblioStats.__new__(BiblioStats)  # Bypass __init__
    BiblioStats.__init__(temp, *args, **kwargs)  # Call __init__ manually
    return temp

class BiblioGroup:
    def __init__(self, f_name=None, db="", df=None, group_desc=None, 
                 res_folder="results-groups", 
                 output_lang="en", preprocess_level=0,
                 exclude_list_kw=None, synonyms_kw=None, lemmatize_kw=False,
                 default_keywords="author", lang_of_docs="en", fancy_output=False, label_docs=True,
                 group_colors=None,
                 **kwargs):

        # Create and initialize a temp BiblioStats-like object
        temp = initialize_biblio_common(
            f_name=f_name, db=db, df=df,
            res_folder=res_folder,
            output_lang=output_lang, preprocess_level=preprocess_level,
            exclude_list_kw=exclude_list_kw, synonyms_kw=synonyms_kw,
            lemmatize_kw=lemmatize_kw, default_keywords=default_keywords,
            lang_of_docs=lang_of_docs, fancy_output=fancy_output)

        # Copy all attributes
        self.__dict__.update(temp.__dict__)

        # Group-specific additions
        self.group_desc = group_desc
        
        self.build_groups(**kwargs)
        self.groups, self.group_df = {}, {}
        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name]
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name] = BiblioStats(df=self.group_df[group_name], db=self.db, preprocess_level=0, label_docs=False, res_folder=None)
        
        if group_colors is None:
            group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
            "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
            "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"]
            self.group_colors = {group: group_colors[i] for i, group in enumerate(self.groups.keys())}
        else:
            self.group_colors = group_colors
            

    def build_groups(self, **kwargs):
        """
        Generate group_matrix using the stored group_desc and additional arguments.
        For example: cutpoints, n_periods, year_range, text_column, etc.
        """
        if self.df is None:
            raise ValueError("No dataframe (self.df) to build group matrix from.")
        if self.group_desc is None:
            raise ValueError("No group descriptor (self.group_desc) provided.")
        
        self.group_matrix = utilsbib.generate_group_matrix(
            df=self.df,
            group_desc=self.group_desc,
            **kwargs)
        
    def compare_continuous_vars(self, vrs=["Year", "Cited by", "Entropy", "Sentiment Score"], output_format="long"):
        vrs = [v for v in vrs if v in self.df.columns]
        self.stats_comparison_continous_df = utilsbib.compare_continuous_by_binary_groups(self.df, vrs, self.group_matrix, output_format=output_format)
        
    def get_group_intersections(self, include_ids=True, id_column="Doc ID"):
        if include_ids:
            id_column = self.df[id_column]
        else:
            id_column = None
        self.group_intersections_df = utilsbib.compute_group_intersections(self.group_matrix, include_ids=include_ids, id_column=id_column)

    def process_keywords(self, exclude_list=None, synonyms=None, lemmatize=False):
        self.df = utilsbib.preprocess_keywords(self.df, "Author Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)
        self.df = utilsbib.preprocess_keywords(self.df, "Index Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)
        self.df = utilsbib.preprocess_keywords(self.df, "Author and Index Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)
        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name]
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name].set_data(self.df)

    def process_text_vars(self, stopwords_file=None, lang="en", remove_numbers=True, remove_two_letter_words=True):
        self.df = utilsbib.process_text_column(self.df, "Abstract", stopwords_file=stopwords_file, lang=lang, remove_numbers=remove_numbers, remove_two_letter_words=remove_two_letter_words)
        self.df = utilsbib.process_text_column(self.df, "Title", stopwords_file=stopwords_file, lang=lang, remove_numbers=remove_numbers, remove_two_letter_words=remove_two_letter_words)
        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name]
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name].set_data(self.df)


    def get_main_info(self, include=["descriptives", "performance", "time series"], performance_mode="full", stopwords=None, excluded_sources_references=None):
        main_info = []
        if self.group_desc == "Year" and "time series" in include:
            include.remove("time series")
        for group_name in self.group_matrix.columns:
            self.groups[group_name].get_main_info(include=include, performance_mode=performance_mode, stopwords=stopwords, excluded_sources_references=excluded_sources_references)

        if "descriptives" in include:
            self.descriptives_df = utilsbib.merge_group_performances({group_name: self.groups[group_name].descriptives_df for group_name in self.group_matrix.columns})
            main_info.append((self.descriptives_df, "descriptives"))
        if "performance" in include:
            self.performances_df = utilsbib.merge_group_performances({group_name: self.groups[group_name].performances_df for group_name in self.group_matrix.columns})
            main_info.append((self.performances_df, "performances"))
        if "time series" in include:
            self.time_series_stats_df = utilsbib.merge_group_performances({group_name: self.groups[group_name].time_series_stats_df for group_name in self.group_matrix.columns})
            main_info.append((self.time_series_stats_df, "time-series analysis"))
        if "references" in include:
            self.references_stats_df = utilsbib.merge_group_performances({group_name: self.groups[group_name].references_stats_df for group_name in self.group_matrix.columns})
            main_info.append((self.references_stats_df, "references"))
        if "specific" in include:
            pass
        
        if self.res_folder is not None:
            utilsbib.save_descriptives_to_excel(main_info, self.res_folder + "\\tables\\main info.xlsx") 
            
    def get_group_top_cited_documents(self, top_n=10, cols=None, filters=None, mode="global",
                                      title_col="Title", ref_col="References", cite_col="Cited by"):
        global_frames, local_frames = [], []
    
        for g in self.group_matrix.columns:
            ga = self.groups[g]
            ga.get_top_cited_documents(top_n, cols, filters, mode, title_col, ref_col, cite_col)
    
            if mode in {"global", "both"} and ga.top_cited_docs_global_df is not None:
                df = ga.top_cited_docs_global_df.copy()
                df["Group"] = g
                global_frames.append(df)
    
            if mode in {"local", "both"} and ga.top_cited_docs_local_df is not None:
                df = ga.top_cited_docs_local_df.copy()
                df["Group"] = g
                local_frames.append(df)
    
        if mode in {"global", "both"}:
            self.top_cited_docs_global_group_df = pd.concat(global_frames, ignore_index=True)
            if self.res_folder is not None:
                utilsbib.to_excel_fancy(self.top_cited_docs_global_group_df, self.res_folder + "\\tables\\top cited documents global.xlsx")
    
        if mode in {"local", "both"}:
            self.top_cited_docs_local_group_df = pd.concat(local_frames, ignore_index=True)   
            if self.res_folder is not None:
                utilsbib.to_excel_fancy(self.top_cited_docs_local_group_df, self.res_folder + "\\tables\\top cited documents global.xlsx")
    
    def count_sources(self, merge_type="all items"):
        self.sources_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_sources", merge_type=merge_type)
    
    def count_author_keywords(self, merge_type="all items"):
        self.author_keywords_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_author_keywords", merge_type=merge_type)
            
    def count_index_keywords(self, merge_type="all items"):
        self.index_keywords_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_index_keywords", merge_type=merge_type)
    
    def count_keywords(self, merge_type="all items"):
        self.count_author_keywords(merge_type="all items")
        self.count_index_keywords(merge_type="all items")
        
    def count_ca_countries(self, merge_type="all items"):
        self.ca_countries_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_ca_countries", merge_type=merge_type)
    
    
    #def count_authors(self, merge_type="all items"):
    #    self.authors_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_authors", merge_type=merge_type)
    
    def count_affiliations(self, merge_type="all items"):
        self.affiliations_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_affiliations", merge_type=merge_type)
        
    def count_references(self, merge_type="all items"):
        self.references_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_references", merge_type=merge_type)
      
    def count_ngrams_abstract(self, merge_type="all items", **kwargs):
        self.words_abs_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_ngrams_abstract", merge_type=merge_type, **kwargs)
        
    def count_ngrams_title(self, merge_type="all items", **kwargs):
        self.words_tit_counts_df = utilsbib.count_occurrences_across_groups(self.groups, self.group_matrix, "count_ngrams_title", merge_type=merge_type, **kwargs)
        
    def count_ngrams(self, merge_type="all items", **kwargs):
        self.count_ngrams_abstract(merge_type=merge_type, **kwargs)
        self.count_ngrams_title(merge_type=merge_type, **kwargs)
        
    def get_sentiment(self, v="Processed Abstract", sentiment_threshold=0.05, top_words=10):
        if v not in self.df.columns:
            if v.split()[1] in self.df.columns:
                self.process_text_vars()
            else:
                print("Text variable not present")
                return None                
        self.df, self.sentiment_stats_df = utilsbib.analyze_sentiment(self.df, v, sentiment_threshold=sentiment_threshold, top_words=top_words)
                
        
    def get_scientific_production(self, relative_counts=True, cumulative=True, predict_last_year=True, percent_change=True, output_format="wide"):
        
        self.production_df = utilsbib.get_scientific_production_by_group(
            self.df, self.group_matrix, relative_counts=relative_counts, cumulative=cumulative,
            predict_last_year=predict_last_year, percent_change=percent_change,
            output_format=output_format)



def get_groups_by_clustering(df, db, preprocess_keywords=False, preprocess_text_vars=True, 
                             text_field="Abstract", method="kmeans", n_clusters=None,
                             k_range=range(2,11), coupling_fields=None, **kwargs):
    ba = BiblioStats(df=df, db=db)
    if preprocess_keywords:
        ba.process_keywords(**kwargs)
    if preprocess_text_vars:
        ba.process_text_vars(**kwargs)
    text_field = [f for f in ba.df.columns if f in ["Processed "+text_field, text_field]][0]
    ba.cluster_documents(text_field=text_field, method=method, n_clusters=n_clusters,
                          k_range=k_range, coupling_fields=coupling_fields)
    return BiblioGroup(df=ba.df, db=db, group_desc=ba.new_column)
    
    
   