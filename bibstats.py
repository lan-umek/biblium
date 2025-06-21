# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:50:08 2025

@author: Lan.Umek
"""

import readbib, utilsbib, reportbib
import pandas as pd
import numpy as np
import networkx as nx

#import sandbox

import os

class BiblioStats:
    
    def __init__(self, f_name=None, db="", df=None, res_folder="results",
                 output_lang="en", preprocess_level=0, exclude_list_kw=None,
                 synonyms_kw=None, lemmatize_kw=False, default_keywords="author",
                 asjc_map_df=None, lang_of_docs="en", fancy_output=False, label_docs=True,
                 dpi=600, cmap="viridis", cmap_disc="tab10", default_color="lightblue"):
        
        self.db = db.lower()
        self.ldf = lambda x: utilsbib.ldf(x,  l=output_lang)
        if f_name is not None:
            self.df = readbib.read_bibfile(f_name, self.db)
        elif df is not None:
            self.df = df
        else:
            print(self.ldf("No dataset has been provided."),  
                  self.ldf("The bibliometric analysis cannot be performed."))
            return None
        
        self.n = len(self.df)
        if label_docs:
            self.df["Doc ID"] = [f"Doc {i}" for i in range(1, self.n+1)]
        
        # mapping dictionaries
        if "Source title" in self.df.columns and "Abbreviated Source Title" in self.df.columns:
            self.sources_abb_dict = self.df.set_index("Source title")[ "Abbreviated Source Title"].to_dict()
            
        if res_folder is not None:
            self.res_folder = os.getcwd() + "\\" + res_folder
            sub_folders = ["plots", "tables", "reports", "networks"]
            folders = [self.res_folder + "\\" + s for s in sub_folders]
            utilsbib.make_folders(folders)
            if fancy_output:
                self.cond_formatting, self.autofit = True, True
            else:
                self.cond_formatting, self.autofit = False, False     
        else:
            self.res_folder = None

        if preprocess_level >= 1:
            self.df = utilsbib.add_ca_country_df(self.df, self.db)
            self.missings_df, self.missings = utilsbib.check_missing_values(self.df)
            self.df = utilsbib.add_document_labels_abbrev(self.df)
            self.id_short_label_dict, self.id_label_dict = self.df.set_index("Doc ID").to_dict()["Document Short Label"], self.df.set_index("Doc ID").to_dict()["Document Label"]
                 
        if preprocess_level >= 2:
            self.process_keywords(exclude_list=exclude_list_kw, synonyms=synonyms_kw, lemmatize=lemmatize_kw)  
            self.process_text_vars(stopwords_file=utilsbib.stopwords_file, lang=lang_of_docs)
            self.get_country_collaboration()
        if preprocess_level >= 3:
            if self.db == "scopus":
                cited_sciences = (preprocess_level == 4)
                self.add_sciences_scopus(asjc_map_df=asjc_map_df, cited_sciences=cited_sciences)
            self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(self.df, author_col="Author Keywords", index_col="Index Keywords")
            self.df = utilsbib.merge_text_columns(self.df, title_col = "Processed Title",  abstract_col = "Processed Abstract", author_col = "Processed Author Keywords", index_col = "Processed Index Keywords")
        if preprocess_level >= 4:
            if hasattr(self, "cited_sciences_df"):
                self.compute_interdisciplinarity_entropy()

            pass
        
        self.describe_columns()
        
        if default_keywords == "author":
            self.kw_var = "Processed Author Keywords" if "Processed Author Keywords" in self.df.columns else "Author Keywords"
        elif default_keywords == "index":
            self.kw_var = "Processed Index Keywords" if "Processed Index Keywords" in self.df.columns else "Index Keywords"
        else:
            self.kw_var = None
            
        self.dpi = dpi
        self.cmap, self.default_color, self.cmap_disc = cmap, default_color, cmap_disc
        
        if "Author full names" in self.df.columns:
            self.id_to_author, self.author_to_id = utilsbib.extract_author_mappings(self.df, "Author full names")
        
        
        if self.res_folder is not None:
            if hasattr(self, "missings_df"):
                utilsbib.to_excel_fancy(self.missings_df, f_name=self.res_folder + "\\tables\\missing values.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_country_collaboration(self, normalization="jaccard", links_df=True, min_weight=1):
         if self.db == "scopus":
             aff_column = "Affiliations"
         self.df, self.country_collab_matrix = utilsbib.extract_countries_from_affiliations(self.df, aff_column=aff_column)
         if normalization is not None:
             self.country_collab_matrix_norm = utilsbib.normalize_symmetric_matrix(self.country_collab_matrix, method=normalization)
         if links_df:
             self.countries_links_df = utilsbib.build_links_from_matrix(self.country_collab_matrix, min_weight=min_weight)
             self.countries_links_df["source"] = self.countries_links_df["source"].map(utilsbib.country_iso3_dct)
             self.countries_links_df["target"] = self.countries_links_df["target"].map(utilsbib.country_iso3_dct)

    def compute_interdisciplinarity_entropy(self, counting_types=["Number of documents", "Proportion of documents", "Franctional number of documents"], concat=True):
        self.entropies_df = utilsbib.compute_interdisciplinarity_entropy(self.cited_sciences_df, counting_types=counting_types)
        if concat:
            self.df = pd.concat([self.df, self.entropies_df.reset_index(drop=True)], axis=1)
        

    def describe_columns(self, show=False):
        """
        Opens the Excel file of variable names and descriptions, 
        prints out each column in `df` alongside its description, 
        and returns a function to look up individual variable descriptions.
        
        Parameters:
        - mapping_file: Path to the Excel file containing 'Name' and 'Description' columns.
        - df: DataFrame whose columns you want to describe.
        
        Returns:
        - get_description: function that takes a variable name and returns its description.
        """
        # Load the mapping of names → descriptions
        fd = os.path.dirname(__file__)
        mapping_df = pd.read_excel(fd+"\\additional files\\variable names.xlsx", sheet_name="descriptions", usecols=["Name", "Description"])
        mapping_dict = dict(zip(mapping_df["Name"], mapping_df["Description"]))
    
        # Display each column with its description
        for col in self.df.columns:
            desc = mapping_dict.get(col, "No description available")
            if show:
                print(f"{col}: {desc}")
    
        # Return a helper function for individual lookups
        def get_description(var_name: str) -> str:
            return mapping_dict.get(var_name, "No description available")
    
        self.column_descriptor = get_description

# Example usage:
# df = pd.read_csv("your_data.csv")
# get_desc = load_and_describe("scopus_variables.xlsx", df)
# print(get_desc("Title"))  # prints the description for the "Title" variable.

    
    def add_sciences_scopus(self, asjc_map_df=None, cited_sciences=True):
        
        fd = os.path.dirname(__file__)        
        if asjc_map_df is None:
            try:
                asjc_map_df = pd.read_excel(fd + "\\additional files\\sources_data_short.xlsx")
            except:
                try:
                    asjc_map_df = pd.read_excel(fd + "\\additional files\\sources_data.xlsx", sheet_name="Scopus Sources Oct. 2024")
                    asjc_map_df = asjc_map_df[["Source Title", "All Science Journal Classification Codes (ASJC)"]]
                except:
                    pass
        asjc_meta_df = pd.read_excel(fd + "\\additional files\\scopus subject area codes.xlsx")
        self.df = utilsbib.enrich_bibliometric_data(self.df, asjc_map_df, asjc_meta_df)
        if cited_sciences:
            self.cited_sciences_df = utilsbib.extract_cited_sciences(self.df, asjc_map_df, asjc_meta_df)
                      
    def process_keywords(self, exclude_list=None, synonyms=None, lemmatize=False):
        self.df = utilsbib.preprocess_keywords(self.df, "Author Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)
        self.df = utilsbib.preprocess_keywords(self.df, "Index Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)
        self.df = utilsbib.preprocess_keywords(self.df, "Author and Index Keywords", exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize)


    def process_text_vars(self, stopwords_file=None, lang="en", remove_numbers=True, remove_two_letter_words=True):
        self.df = utilsbib.process_text_column(self.df, "Abstract", stopwords_file=stopwords_file, lang=lang, remove_numbers=remove_numbers, remove_two_letter_words=remove_two_letter_words)
        self.df = utilsbib.process_text_column(self.df, "Title", stopwords_file=stopwords_file, lang=lang, remove_numbers=remove_numbers, remove_two_letter_words=remove_two_letter_words)
        
    def get_production(self, relative_counts=True, cumulative=True, predict_last_year=False):
        self.production_df = utilsbib.get_scientific_production(self.df, relative_counts=relative_counts, cumulative=cumulative, predict_last_year=predict_last_year)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.production_df, f_name=self.res_folder + "\\tables\\scientific production.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)
    
    # Descriptives

    def get_collaboration_index(self, author_col="Author(s) ID", sep=";"):
        self.collaboration_index = utilsbib.collaboration_index(self.df, author_col=author_col, sep=sep)
    
    def get_main_info(self, include=["descriptives", "performance", "time series"], performance_mode="full", stopwords=None, excluded_sources_references=None):
        main_info = []
        if "descriptives" in include:
            self.descriptives_df = utilsbib.compute_descriptive_statistics(
            self.df, [("Year", "numeric"), ("Source title", "string"), ("Document Type", "string"),
                      ("Open Access", "string"),  ("Cited by", "numeric"), ("Author Keywords", "list"),
                      ("Index Keywords", "list"), ("Language of Original Document", "string"),
                      ("Abstract", "text"), ("Title", "text")], stopwords=stopwords)
            main_info.append((self.descriptives_df, "descriptives"))
        if "performance" in include:
            metrics = utilsbib.get_performance_indicators(self.df, mode=performance_mode)
            data = [("Performance indicator", name, value) for name, value in metrics]
            self.performances_df = pd.DataFrame(data, columns=["Variable", "Indicator", "Value"])
            main_info.append((self.performances_df, "performances"))
        if "time series" in include:
            if not hasattr(self, "production_df"):
                self.get_production()
            self.time_series_stats_df = utilsbib.summarize_publication_timeseries(self.production_df)
            main_info.append((self.time_series_stats_df, "time-series analysis"))
        if "references" in include:
            if not hasattr(self, "df_references"):
                self.df_references = utilsbib.parse_references_dataframe(self.df, excluded_sources=excluded_sources_references)
            self.references_stats_df = utilsbib.summarize_parsed_references(self.df_references)
            main_info.append((self.references_stats_df, "references"))
            
        if "specific" in include:
            sent = utilsbib.get_specific_indicators(self.df)
            data = [("Sentiment analysis", name, value) for name, value in sent]
            self.specific_stats_df = pd.DataFrame(data, columns=["Variable", "Indicator", "Value"])

            main_info.append((self.specific_stats_df, "specifics"))
        
        if self.res_folder is not None:
            utilsbib.save_descriptives_to_excel(main_info, self.res_folder + "\\tables\\main info.xlsx")
    
    # Top cited documents
    
    def get_top_cited_documents(self, top_n=10, cols=None, filters=None, mode="global",
                                 title_col="Title", ref_col="References", cite_col="Cited by"):
        """
        Compute and store top-cited documents (global, local, or both), using external helper functions.
        """
        if mode in {"global", "both"}:
            self.top_cited_docs_global_df = utilsbib.select_global_top_cited_documents(
                self.df, top_n=top_n, cols=cols, filters=filters, cite_col=cite_col)
            if self.res_folder is not None:
                utilsbib.to_excel_fancy(self.top_cited_docs_global_df, self.res_folder + "\\tables\\top cited documents global.xlsx")
        if mode in {"local", "both"}:
            self.top_cited_docs_local_df = utilsbib.select_local_top_cited_documents(
                self.df, top_n=top_n, cols=cols, filters=filters,
                title_col=title_col, ref_col=ref_col, cite_col=cite_col)
            if self.res_folder is not None:
                utilsbib.to_excel_fancy(self.top_cited_docs_local_df, self.res_folder + "\\tables\\top cited documents local.xlsx")   
    
    # Counting
            
    # single occurecenes
    def count_sources(self, top_n=0, **kwargs):
        self.sources_counts_df = utilsbib.count_occurrences(
            self.df, "Source title", count_type="single",
            item_column_name="Source",
            rename_dict=self.sources_abb_dict,
            translated_column_name="Abbreviated Source Title"
        )
        if top_n > 0:
            top_items = self.sources_counts_df["Source"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Source title",
                items_of_interest=top_items,
                value_type="string", **kwargs
            )
            self.binary_sources_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.sources_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "sources counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.sources_counts_df
    
    def count_document_types(self, top_n=0, **kwargs):
        self.document_types_counts_df = utilsbib.count_occurrences(
            self.df, "Document Type", count_type="single", item_column_name="Document Type"
        )
        if top_n > 0:
            top_items = self.document_types_counts_df["Document Type"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Document Type",
                items_of_interest=top_items,
                value_type="string", **kwargs
            )
            self.binary_document_types_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.document_types_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "document types counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.document_types_counts_df
    
    def count_ca_countries(self, top_n=0, **kwargs):
        if "CA Country" not in self.df.columns:
            self.df = utilsbib.add_ca_country_df(self.df, self.db)
        self.ca_country_counts_df = utilsbib.count_occurrences(
            self.df, "CA Country", count_type="single", item_column_name="Country"
        )
        self.ca_country_counts_df["ISO-3"] = self.ca_country_counts_df["Country"].map(utilsbib.country_iso3_dct)
        self.ca_country_counts_df["Continent"] = self.ca_country_counts_df["Country"].map(utilsbib.continent_dct)
        if top_n > 0:
            top_items = self.ca_country_counts_df["Country"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="CA Country",
                items_of_interest=top_items,
                value_type="string", **kwargs
            )
            self.binary_ca_countries_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.ca_country_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "CA country counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.ca_country_counts_df
    
    def count_author_keywords(self, top_n=0, **kwargs):
        v = "Processed Author Keywords" if "Processed Author Keywords" in self.df.columns else "Author Keywords"
        self.author_keywords_counts_df = utilsbib.count_occurrences(
            self.df, v, count_type="list", item_column_name="Keyword"
        )
        if top_n > 0:
            top_items = self.author_keywords_counts_df["Keyword"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=v,
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_author_keywords_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.author_keywords_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "author keywords counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.author_keywords_counts_df
    
    def count_index_keywords(self, top_n=0, **kwargs):
        v = "Processed Index Keywords" if "Processed Index Keywords" in self.df.columns else "Index Keywords"
        self.index_keywords_counts_df = utilsbib.count_occurrences(
            self.df, v, count_type="list", item_column_name="Keyword"
        )
        if top_n > 0:
            top_items = self.index_keywords_counts_df["Keyword"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=v,
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_index_keywords_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.index_keywords_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "index keywords counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.index_keywords_counts_df
    
    def count_keywords(self, which=None, top_n=0, **kwargs):
        kw_var = self.kw_var if which is None else which
        self.keywords_counts_df = utilsbib.count_occurrences(
            self.df, kw_var, count_type="list", item_column_name="Keyword"
        )
        if top_n > 0:
            top_items = self.keywords_counts_df["Keyword"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=kw_var,
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_keywords_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.keywords_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "keywords counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.keywords_counts_df
    
    def count_authors(self, top_n=0, **kwargs):
        self.author_var = [v for v in ["Author full names", "Author(s) ID", "Authors"] if v in self.df.columns][0]
        self.authors_counts_df = utilsbib.count_occurrences(
            self.df, self.author_var, count_type="list", item_column_name="Author ID"
        )
        if "Author full names" in self.df.columns:
            self.authors_counts_df = utilsbib.split_author_id(self.authors_counts_df)
        if top_n > 0:
            top_items = self.authors_counts_df["Author ID"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=self.author_var,
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_authors_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.authors_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "authors counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.authors_counts_df
    
    def count_affiliations(self, top_n=0, **kwargs):
        self.aff_counts_df = utilsbib.count_occurrences(
            self.df, "Affiliations", count_type="list", item_column_name="Affiliation"
        )
        if top_n > 0:
            top_items = self.aff_counts_df["Affiliation"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Affiliations",
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_affiliations_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.aff_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "affiliation counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.aff_counts_df
    
    def count_fields(self, top_n=0, **kwargs):
        self.fields_counts_df = utilsbib.count_occurrences(
            self.df, "Field", count_type="list", item_column_name="Field", sep=";"
        )
        if top_n > 0:
            top_items = self.fields_counts_df["Field"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Field",
                items_of_interest=top_items,
                value_type="list",
                separator=";", **kwargs
            )
            self.binary_fields_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.fields_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "fields counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.fields_counts_df
    
    def count_areas(self, top_n=0, **kwargs):
        self.areas_counts_df = utilsbib.count_occurrences(
            self.df, "Area", count_type="list", item_column_name="Area", sep=";"
        )
        if top_n > 0:
            top_items = self.areas_counts_df["Area"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Area",
                items_of_interest=top_items,
                value_type="list",
                separator=";", **kwargs
            )
            self.binary_areas_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.areas_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "areas counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.areas_counts_df
    
    def count_sciences(self, top_n=0, **kwargs):
        self.sciences_counts_df = utilsbib.count_occurrences(
            self.df, "Science", count_type="list", item_column_name="Science", sep=";"
        )
        if top_n > 0:
            top_items = self.sciences_counts_df["Science"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Science",
                items_of_interest=top_items,
                value_type="list",
                separator=";", **kwargs
            )
            self.binary_sciences_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.sciences_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "sciences counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.sciences_counts_df
    
    # Add new or update existing functions:
    def count_references(self, min_len=50, top_n=0, **kwargs):
        self.references_counts_df = utilsbib.count_occurrences(
            self.df, "References", count_type="list", item_column_name="Reference", sep=";"
        )
        self.references_counts_df = self.references_counts_df[self.references_counts_df["Reference"].str.len() >= min_len]
        if top_n > 0:
            top_items = self.references_counts_df["Reference"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="References",
                items_of_interest=top_items,
                value_type="list",
                separator=";", **kwargs
            )
            self.binary_references_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.references_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "references counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.references_counts_df
    
    def count_ngrams_abstract(self, ngram_range=(1, 2), top_n=0, **kwargs):
        v = "Processed Abstract" if "Processed Abstract" in self.df.columns else "Abstract"
        self.words_abs_counts_df = utilsbib.count_occurrences(
            self.df, v, count_type="text", item_column_name="Word - Phrase", ngram_range=ngram_range
        )
        if top_n > 0:
            top_items = self.words_abs_counts_df["Word - Phrase"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=v,
                items_of_interest=top_items,
                value_type="text", **kwargs
            )
            self.binary_words_abs_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.words_abs_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "words abstract counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.words_abs_counts_df
    
    def count_ngrams_title(self, ngram_range=(1, 2), top_n=0, **kwargs):
        v = "Processed Title" if "Processed Title" in self.df.columns else "Title"
        self.words_tit_counts_df = utilsbib.count_occurrences(
            self.df, v, count_type="text", item_column_name="Word - Phrase", ngram_range=ngram_range
        )
        if top_n > 0:
            top_items = self.words_tit_counts_df["Word - Phrase"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col=v,
                items_of_interest=top_items,
                value_type="text", **kwargs
            )
            self.binary_words_tit_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.words_tit_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "words tit counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        return self.words_tit_counts_df
    
    def count_ngrams(self, ngram_range=(1, 2), top_n=0, **kwargs):
        self.count_ngrams_abstract(ngram_range=ngram_range, top_n=top_n, **kwargs)
        self.count_ngrams_title(ngram_range=ngram_range, top_n=top_n, **kwargs)
    
    def count_all_countries(self, top_n=0, **kwargs):
        self.all_countries_counts_df = utilsbib.count_occurrences(
            self.df, "Countries of Authors", count_type="list", item_column_name="Country"
        )
        if top_n > 0:
            top_items = self.all_countries_counts_df["Country"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Countries of Authors",
                items_of_interest=top_items,
                value_type="list", **kwargs
            )
            self.binary_all_countries_df = indicators_dict["binary"]
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(
                self.all_countries_counts_df,
                f_name=os.path.join(self.res_folder, "tables", "all countries counts.xlsx"),
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting
            )
        self.all_countries_counts_df["ISO-3"] = self.all_countries_counts_df["Country"].map(utilsbib.country_iso3_dct)
        return self.all_countries_counts_df
    
    def count_all(self, top_n=0, **kwargs):
        for f in [
            "count_sources", "count_document_types", "count_ca_countries", "count_author_keywords",
            "count_index_keywords", "count_authors", "count_affiliations",
            "count_references", "count_fields", "count_areas", "count_sciences",
            "count_ngrams_abstract", "count_ngrams_title", "count_all_countries"
        ]:
            try:
                getattr(self, f)(top_n=top_n, **kwargs)
            except Exception:
                print("Problem", f)
                pass

    
    
    # performance measuring

    """
    Compute performance statistics for a specific entity type (e.g., Source, Author, Country, Document Type).
    This method selects relevant items based on provided filters or top-N counts and delegates the analysis
    to a generic utility function `get_entity_stats`.
    
    The resulting statistics are stored in a DataFrame attribute (e.g., self.source_stats_df),
    and if `indicators=True`, a corresponding indicator matrix is also stored (e.g., self.source_indicators).
    
    Parameters are passed through to `get_entity_stats`, including:
    - items_of_interest: list of entities to include explicitly
    - exclude_items: list of entities to exclude
    - top_n: number of top entities to include if items_of_interest is not provided
    - regex_include / regex_exclude: optional patterns to filter entities
    - indicators: whether to return a binary document-entity indicator matrix
    - missing_as_zero: whether to treat missing entries as zeros
    - mode: analysis mode passed to performance metric function
    
    Relies on a user-defined `count_<entity>` method if counts_df is not provided.
    """

    def get_sources_stats(self, **kwargs):
        self.sources_stats_df, self.sources_indcators = utilsbib.get_entity_stats(
            self.df, "Source title", "Source",
            count_method=self.count_sources, value_type="string", **kwargs)
        self.sources_stats_df["Abbreviated Source Title"] = self.sources_stats_df["Source"].map(self.sources_abb_dict)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.sources_stats_df, f_name=self.res_folder + "\\tables\\sources stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)
                    
    def get_document_type_stats(self, **kwargs):
        self.document_type_stats_df, self.document_type_indcators = utilsbib.get_entity_stats(
            self.df, "Document Type", "Document Type",
            count_method=self.count_document_types, value_type="string", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.document_type_stats_df, f_name=self.res_folder + "\\tables\\document types stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_ca_countries_stats(self, **kwargs):
        self.ca_countries_stats_df, self.ca_countries_indcators = utilsbib.get_entity_stats(
                self.df, "CA Country", "Country",
                count_method=self.count_ca_countries, value_type="string", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.ca_countries_stats_df, f_name=self.res_folder + "\\tables\\ca countries stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_author_keywords_stats(self, **kwargs):
        keyword_column = "Processed Author Keywords" if "Processed Author Keywords" in self.df.columns else "Author Keywords"
        self.author_keywords_stats_df, self.author_keywords_indcators = utilsbib.get_entity_stats(
                self.df, keyword_column, "Keyword",
                count_method=self.count_author_keywords, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.author_keywords_stats_df, f_name=self.res_folder + "\\tables\\author keywords stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_index_keywords_stats(self, **kwargs):
        keyword_column = "Processed Index Keywords" if "Processed Index Keywords" in self.df.columns else "Index Keywords"
        self.index_keywords_stats_df, self.index_keywords_indcators = utilsbib.get_entity_stats(
                self.df, keyword_column, "Keyword",
                count_method=self.count_index_keywords, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.index_keywords_stats_df, f_name=self.res_folder + "\\tables\\index keywords stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)
        
    def get_keywords_stats(self, keyword_types=["author", "index"], **kwargs):
        for keyword_type in keyword_types:
            if keyword_type == "author":
                self.get_author_keywords_stats(**kwargs)
            elif keyword_type == "index":
                self.get_index_keywords_stats(**kwargs)
                
    def get_authors_stats(self, **kwargs): # ne dela OK
        self.authors_stats_df, self.authos_indcators = utilsbib.get_entity_stats(
                self.df, "Author full names", "Author ID",
                count_method=self.count_authors, value_type="list", **kwargs)
        if "Author full names" in self.df.columns:
            self.authors_stats_df = utilsbib.split_author_id(self.authors_stats_df) 
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.authors_stats_df, f_name=self.res_folder + "\\tables\\author stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_fields_stats(self, **kwargs): 
        self.fields_stats_df, self.fields_indcators = utilsbib.get_entity_stats(
                self.df, "Field", "Field",
                count_method=self.count_fields, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.fields_stats_df, f_name=self.res_folder + "\\tables\\fields stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_areas_stats(self, **kwargs): 
        self.areas_stats_df, self.areas_indcators = utilsbib.get_entity_stats(
                self.df, "Area", "Area",
                count_method=self.count_areas, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.areas_stats_df, f_name=self.res_folder + "\\tables\\areas stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_sciences_stats(self, **kwargs): 
        self.sciences_stats_df, self.sciences_indcators = utilsbib.get_entity_stats(
                self.df, "Science", "Science",
                count_method=self.count_sciences, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.sciences_stats_df, f_name=self.res_folder + "\\tables\\sciences stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)
    
    def get_all_sciences_stats(self, **kwargs):
        self.get_fields_stats(**kwargs)
        self.get_areas_stats(**kwargs)
        self.get_sciences_stats(**kwargs)
    
    def get_affiliations_stats(self, **kwargs): 
        self.affiliations_stats_df, self.affiliations_indcators = utilsbib.get_entity_stats(
                self.df, "Affiliations", "Affiliation",
                count_method=self.count_affiliations, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.affiliations_stats_df, f_name=self.res_folder + "\\tables\\affiliations stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_ngrams_abstract_stats(self, **kwargs): 
        v = "Processed Abstract" if "Processed Abstract" in self.df.columns else "Abstract"
        self.ngrams_abstract_stats_df, self.ngrams_abstract_stats_indicators = utilsbib.get_entity_stats(
                self.df, v, "Word - Phrase",
                count_method=self.count_ngrams_abstract, value_type="text", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.ngrams_abstract_stats_df, f_name=self.res_folder + "\\tables\\ngrams abstract stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_ngrams_title_stats(self, **kwargs): 
        v = "Processed Title" if "Processed Title" in self.df.columns else "Title"
        self.ngrams_title_stats_df, self.ngrams_title_stats_indicators = utilsbib.get_entity_stats(
                self.df, v, "Word - Phrase",
                count_method=self.count_ngrams_title, value_type="text", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.ngrams_title_stats_df, f_name=self.res_folder + "\\tables\\ngrams title stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_references_stats(self, **kwargs):
        self.references_stats_df, self.references_stats_indicators = utilsbib.get_entity_stats(
                self.df, "References", "Reference",
                count_method=self.count_references, value_type="list", **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.references_stats_df, f_name=self.res_folder + "\\tables\\references stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    def get_all_countries_stats(self, top_n=200, **kwargs):
        self.all_countries_stats_df, self.all_countries_indcators = utilsbib.get_entity_stats(
                self.df, "Countries of Authors", "Country",
                count_method=self.count_all_countries, value_type="list", top_n=top_n, **kwargs)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.all_countries_stats_df, f_name=self.res_folder + "\\tables\\all countries stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    
    def get_all_items_stats(self, **kwargs):
        for f in ["get_sources_stats", "get_document_types_stats", "get_ca_countries_stats",
                  "get_author_keywords_stats", "get_index_keywords_stats", "get_authors_stats",
                  "get_affiliations_stats", "get_references_stats", "get_fields_stats", "get_areas_stats",
                  "get_sciences_stats", "get_ngrams_abstract_stats", "get_ngrams_title_stats"]:
            try:
                getattr(self, f)(**kwargs)
            except:
                pass
            
    def compute_reference_spectrogram(self, reference_column="References", include_pre_1900=False):
        self.spectrogram_df = utilsbib.compute_reference_spectrogram(self.df, reference_column=reference_column, include_pre_1900=include_pre_1900)
        

    # factor analysis
    
    def conceptual_structure_analysis(self,
        field: str = "Author Keywords",
        dr_method: str = "MCA",
        cluster_method: str = "kmeans",
        n_clusters: int = 5,
        n_terms: int = 100,
        n_components: int = 2,
        dtm_method: str = "count",
        term_selection: str = "frequency",
        y: np.ndarray = None,
        min_df: int = 2,
        ngram_range: tuple = (1, 1),
        use_lemmatization: bool = False,
        pos_filter: list = None,
        include_terms: list = None,
        exclude_terms: list = None,
        term_regex: str = None,
        compute_metrics: bool = False):
        
        field = [f for f in [f"Processed {field}", field] if f in self.df.columns][0]
        print(field)
        
        self.conceptual_structure_d = utilsbib.conceptual_structure_analysis(
            self.df, field=field, dr_method=dr_method, cluster_method=cluster_method,
            n_clusters=n_clusters, n_terms=n_terms, dtm_method=dtm_method,
            term_selection=term_selection, y=y, min_df=min_df, ngram_range=ngram_range,
            use_lemmatization=use_lemmatization, pos_filter=pos_filter,
            include_terms=include_terms, exclude_terms=exclude_terms,
            term_regex=term_regex, compute_metrics=compute_metrics)
        
        self.words_by_cluster_df = utilsbib.words_by_cluster(
            self.conceptual_structure_d["term_embeddings"],
            self.conceptual_structure_d["terms"],
            self.conceptual_structure_d["term_labels"])
        
        
    


    # relations computation
    
    # general coocurences
    

    def compute_cooccurrence(
            self, column_name, items_of_interest=None, top_n=20, normalization=False, network=True,
            partition_network=True, partition_kwargs=None,
            vector_df=None, vector_cols=None,
            network_filename=None, network_formats=["pajek"],
            top_items_df=None, top_items_col=None, count_func=None, count_attr=None,
            output_attr_prefix=None, value_type="list", separator="; ", **kwargs):
        """
        Compute the co-occurrence matrix and, optionally, normalized matrices and a network graph
        for items in a DataFrame column. Optionally adds partitions, node vector attributes, and exports the network.
    
        Args:
            column_name (str): Name of the DataFrame column containing items.
            items_of_interest (list, optional): Items to include in the analysis.
            top_n (int): If items_of_interest is not provided, selects the top N items.
            normalization (bool): If True, also computes normalized co-occurrence matrices.
            network (bool): If True, computes and returns a networkx graph of co-occurrences.
            partition_network (bool): If True (default), adds partitions to the network using utilsbib.add_partitions.
            partition_kwargs (dict, optional): Keyword arguments passed to utilsbib.add_partitions.
            vector_df (pd.DataFrame, optional): DataFrame containing node vectors to add as node attributes.
            vector_cols (list of str, optional): Names of columns in vector_df to use as node vector attributes.
            network_filename (str, optional): Prefix or path for saving the network in various formats.
            network_formats (list of str, optional): List of formats to save ("pajek", "graphml", "gexf").
            top_items_df (pd.DataFrame, optional): Precomputed DataFrame of top items.
            top_items_col (str, optional): Column name in top_items_df for item names.
            count_func (callable, optional): Function to compute top items if needed.
            count_attr (str, optional): Attribute name to retrieve computed top items DataFrame.
            output_attr_prefix (str, optional): If provided, stores outputs as instance attributes with this prefix.
            value_type (str): Type of values in the column ("list", etc.).
            separator (str): If value_type is "string", specifies separator to split items.
            **kwargs: Additional arguments passed to select_documents.
    
        Returns:
            tuple: (co-occurrence matrix, dict of normalized matrices, networkx.Graph or None)
        """
        partitions, vectors = {}, {}
        if column_name not in self.df.columns:
            raise ValueError(f"Column \"{column_name}\" not in DataFrame.")
    
        if items_of_interest is None and top_items_df is None and count_func and count_attr:
            count_func()
            top_items_df = getattr(self, count_attr)
    
        _, indicators = utilsbib.select_documents(
            self.df, column_name, value_type=value_type, indicators=True,
            items_of_interest=items_of_interest, top_items_df=top_items_df,
            top_items_col=top_items_col or column_name, top_n=top_n,
            separator=separator, **kwargs
        )
    
        binary_matrix = indicators["binary"]
        
        co_matrix, co_matrices_norm, co_network, all_measures_df, all_measures_df_T = utilsbib.compute_relation_matrix(
            binary_matrix, normalization=normalization, network=network
        )
        
        co_network.remove_edges_from(nx.selfloop_edges(co_network))
    
        # Add partitions to the network if requested and if the network exists
        if partition_network and (co_network is not None):
            partitions = utilsbib.add_partitions(co_network, **(partition_kwargs or {}))
    
        # Add vector attributes to nodes if requested and if the network exists
        if co_network is not None and vector_df is not None and vector_cols is not None:
            vector_cols = [v for v in vector_cols if v in vector_df.columns]
            node_col = top_items_col or column_name
            co_network = utilsbib.add_vectors_from_dataframe(co_network, vector_df, node_col=node_col, vector_cols=vector_cols)
    
            nodes_to_keep = set(vector_df[node_col])
            co_network = co_network.subgraph(nodes_to_keep).copy()
    
  
        # Save network in requested formats
        if (co_network is not None) and (network_filename is not None) and (self.res_folder is not None):
            network_filename = os.path.join(self.res_folder, "networks", network_filename)
            utilsbib.save_network(
                co_network,
                network_filename,
                formats=network_formats,
                vector_cols=vector_cols,
                partition_attr="partition",  # adjust if needed
            )
        
        # specifics for co-authorship networks
        try:
            co_network = nx.relabel_nodes(co_network, lambda x: x.split(" (")[0])
        except:
            pass
        
        co_network_no_isolated = co_network.copy()

        # Remove nodes with degree 0
        co_network_no_isolated.remove_nodes_from([n for n, d in co_network.degree() if d == 0])
    

    
        if output_attr_prefix:
            setattr(self, f"{output_attr_prefix}_cooccurrence_matrix", co_matrix)
            setattr(self, f"{output_attr_prefix}_cooccurrence_matrices_normalized", co_matrices_norm)
            setattr(self, f"{output_attr_prefix}_cooccurrence_network", co_network)
            setattr(self, f"{output_attr_prefix}_cooccurrence_network_no_isolated", co_network_no_isolated)
            setattr(self, f"{output_attr_prefix}_all_coocurrences", indicators)
            setattr(self, f"{output_attr_prefix}_partitions", partitions)
            #setattr(self, f"{output_attr_prefix}_vectors", vectors)
    
        return co_matrix, co_matrices_norm, co_network



    # sepcific coocurences
    def get_author_keyword_cooccurrence(self, vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "author_keywords_stats_df") and vec_stats:
            self.get_author_keywords_stats()        
        elif not hasattr(self, "author_keywords_counts_df"):
            self.count_author_keywords()
        vector_df = self.author_keywords_stats_df if hasattr(self, "author_keywords_stats_df") else self.author_keywords_counts_df
        vector_cols = [v for v in vector_cols if v in vector_df.columns]
        keyword_col = next((col for col in ["Processed Author Keywords", "Author Keywords"] if col in self.df.columns), None)
        self.compute_cooccurrence(keyword_col, count_func=self.count_author_keywords, count_attr="author_keywords_counts_df", top_items_col="Keyword", output_attr_prefix="ak", network_filename="author keyword coocurrence", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)

    def get_index_keyword_cooccurrence(self, vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "index_keywords_stats_df") and vec_stats:
            self.get_index_keywords_stats()        
        elif not hasattr(self, "index_keywords_counts_df"):
            self.count_index_keywords()
        vector_df = self.index_keywords_stats_df if hasattr(self, "index_keywords_stats_df") else self.index_keywords_counts_df
        keyword_col = next((col for col in ["Processed Index Keywords", "Index Keywords"] if col in self.df.columns), None)
        self.compute_cooccurrence(keyword_col, count_func=self.count_index_keywords, count_attr="index_keywords_counts_df", top_items_col="Keyword", output_attr_prefix="ik", network_filename="index keyword coocurrence",  vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)

    def get_ngrams_title_cooccurrence(self, ngram_range=(1,2), vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "ngrams_title_stats_df") and vec_stats:
            self.get_ngrams_title_stats()               
        elif not hasattr(self, "words_tit_counts_df"):
            self.count_ngrams_title(ngram_range=ngram_range)
        vector_df = self.ngrams_title_stats_df if hasattr(self, "ngrams_title_stats_df") else self.words_tit_counts_df        
        title_col = next((col for col in ["Processed Title", "Title"] if col in self.df.columns), None)
        self.compute_cooccurrence(title_col, count_func=self.count_ngrams_title, count_attr="words_tit_counts_df", top_items_col="Word - Phrase", output_attr_prefix="ngrams_title", value_type="text", network_filename="ngrams title coocurrence", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)

    def get_ngrams_abstract_cooccurrence(self, ngram_range=(1,2), vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "ngrams_abstract_stats_df") and vec_stats:
            self.get_ngrams_abstract_stats()               
        elif not hasattr(self, "words_abs_counts_df"):
            self.count_ngrams_abstract(ngram_range=ngram_range)
        vector_df = self.ngrams_abstract_stats_df if hasattr(self, "ngrams_abstract_stats_df") else self.words_abs_counts_df 
        abstract_col = next((col for col in ["Processed Abstract", "Abstract"] if col in self.df.columns), None)      
        self.compute_cooccurrence(abstract_col, count_func=self.count_ngrams_abstract, count_attr="words_abs_counts_df", top_items_col="Word - Phrase", output_attr_prefix="ngrams_abstract", value_type="text", network_filename="ngrams abstract coocurrence", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)

    def get_co_citations(self, vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "references_stats_df") and vec_stats:
            self.get_references_stats()        
        elif not hasattr(self, "references_counts_df"):
            self.count_references()
        vector_df = self.references_stats_df if hasattr(self, "references_stats_df") else self.references_counts_df

        self.compute_cooccurrence("References", count_func=self.count_references, count_attr="references_counts_df", top_items_col="Reference", output_attr_prefix="refs", value_type="list", network_filename="cocitation", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)
        utilsbib.rename_attributes(self, {"refs_cooccurrence_network": "co_citation_network"}) # tole še dodelaj

    def get_coauthorship(self, vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=20, **kwargs):
        if not hasattr(self, "authors_stats_df") and vec_stats:
            self.get_authors_stats()               
        elif not hasattr(self, "authors_counts_df"):
            self.count_authors()
        vector_df = self.authors_stats_df if hasattr(self, "authors_stats_df") else self.authors_counts_df

        self.compute_cooccurrence(self.author_var, count_func=self.count_authors, count_attr="authors_counts_df", top_items_col="Author ID", output_attr_prefix="auth", value_type="list", network_filename="coauthorship", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)
        utilsbib.rename_attributes(self, {"auth_cooccurrence_network": "co_authorship_network"}) # tole še dodelaj

    def get_country_collaboration_network(self, vec_stats=True, vector_cols=["Number of documents", "Total citations", "H-index", "Average year"], top_n=200, **kwargs):
        if not hasattr(self, "all_countries_stats_df") and vec_stats:
            self.get_all_countries_stats()               
        elif not hasattr(self, "all_countries_counts_df"):
            self.count_all_countries()
        vector_df = self.all_countries_stats_df if hasattr(self, "all_countries_stats_df") else self.all_countries_counts_df

        self.compute_cooccurrence("Countries of Authors", count_func=self.count_all_countries, count_attr="all_countries_counts_df", top_items_col="Country", output_attr_prefix="all_countries", value_type="list", network_filename="country collaboration", vector_df=vector_df, vector_cols=vector_cols, top_n=top_n, **kwargs)
        

    # citation network of documents
    
    def build_citation_network(self, threshold=90, largest_only=True,
                               main_path=True, rename="short"):
        self.citation_network_documents, unmatched = utilsbib.build_citation_network(self.df, threshold=threshold, largest_only=largest_only)
        if main_path:
            self.citation_main_path = utilsbib.compute_main_path(self.citation_network_documents)
        if rename is not None:
            if rename == "short":
                d = self.id_short_label_dict
            else:
                d = self.id_label_dict

            self.citation_network_documents = nx.relabel_nodes(self.citation_network_documents, d)
            self.citation_main_path = [d[doc] for doc in self.citation_main_path]
            
    def build_historiograph(self, title_col="Title", year_col="Year", refs_col="References",
                            cutoff=0.85, label_col="Document Short Label", filename="historiograph"):
        filename = os.path.join(self.res_folder, "networks", filename)
        self.historiograph = utilsbib.build_historiograph(self.df, title_col=title_col,
                                     year_col=year_col, refs_col=refs_col,
                                     cutoff=cutoff, label_col=label_col,
                                     save_path=filename)

    # analysis of relationships
    
    def relate_concepts(self, concept1, concept2, custom_matrices=None, 
                        include_stats=["diversity", "bipartite network", "cluster",
                                       "bicluster", "correspondence",
                                       "chi2", "svd", "log-ratio"], clean_zeros=True,
                        to_self=False, **kwargs):
        known = {
            "Sources": self.sources_indcators["binary"],
            "Author Keywords": self.author_keywords_indcators["binary"],
        }
    
        if concept1 in known:
            df1 = known[concept1]
        elif custom_matrices and concept1 in custom_matrices:
            df1 = custom_matrices[concept1]
        else:
            raise ValueError(f"No binary matrix available for concept1: {concept1}")
    
        if concept2 in known:
            df2 = known[concept2]
        elif custom_matrices and concept2 in custom_matrices:
            df2 = custom_matrices[concept2]
        else:
            raise ValueError(f"No binary matrix available for concept2: {concept2}")

        # Compute relation matrix using existing function
        df_relation = utilsbib.compute_relation_matrix(df1, df2)[0]
        if concept1 == "Sources":
            df_relation = df_relation.rename(index=self.sources_abb_dict)
        if concept2 == "Sources":
            df_relation = df_relation.rename(columns=self.sources_abb_dict)
    
        # Store it
        if not hasattr(self, "relation_matrices"):
            self.relation_matrices = {}
            
        if not hasattr(self, "relations"):
            self.relations = {}
        
        # Forward direction
        if concept1 not in self.relation_matrices:
            self.relation_matrices[concept1] = {}
        self.relation_matrices[concept1][concept2] = df_relation
        
        # Reverse direction (transpose)
        if concept2 not in self.relation_matrices:
            self.relation_matrices[concept2] = {}
        self.relation_matrices[concept2][concept1] = df_relation.T
        
        
        if concept1 not in self.relations:
            self.relations[concept1] = {}
        if concept2 not in self.relations:
            self.relations[concept2] = {}
            
        class Relation:
            def __init__(self, concept1, concept2):
                self.concept1, self.concept2 = concept1, concept2
            def set_matrix(self, rm):
                self.rm = rm
            def link_to_self(self, obj):
                for att in dir(self):
                    name = "Relations_" + concept1.replace(" ", "_") + "_" + concept2.replace(" ", "_") + "_" + att
                    if att[0] != "_":
                        setattr(obj, name, getattr(self, att))
        
        R = Relation(concept1, concept2)
        
        rm = self.relation_matrices[concept1][concept2]
        R.set_matrix(rm)
        
        for stat in include_stats:
            if stat == "diversity":
                metrics = utilsbib.compute_diversity_metrics(rm)
                setattr(R, "diversity_row_metrics", metrics["row_metrics"])
                setattr(R, "diversity_column_metrics", metrics["column_metrics"])
            if stat == "bipartite network":
                bipartite = utilsbib.analyze_bipartite_relation(rm, **kwargs)
                setattr(R, "bipartite_graph", bipartite["bipartite_graph"])
                setattr(R, "bipartite_row_projection", bipartite["row_projection"])
                setattr(R, "bipartite_column_projection", bipartite["column_projection"])
                setattr(R, "bipartite_row_stats", bipartite["row_stats"])
                setattr(R, "bipartite_column_stats", bipartite["column_stats"])
                setattr(R, "bipartite_global_stats", bipartite["bipartite_global"])
                setattr(R, "bipartite_row_global", bipartite["row_global"])
                setattr(R, "bipartite_column_global", bipartite["column_global"])   
            if stat == "cluster":
                result = utilsbib.cluster_relation_matrix(rm, **kwargs)
                setattr(R, "clusters", result["clusters"])
                setattr(R, "n_clusters", result["n_clusters"])
                if "silhouette_scores" in result:
                    setattr(R, "silhouette_scores", result["silhouette_scores"])                
            if stat == "bicluster":
                result = utilsbib.bicluster_relation_matrix(rm, **kwargs)
                setattr(R, "bicluster_model", result["model"])
                setattr(R, "bicluster_row_clusters", result["row_clusters"])
                setattr(R, "bicluster_column_clusters", result["column_clusters"])
            if stat == "correspondence":
                row_coords, col_coords, explained_inertia = utilsbib.compute_correspondence_analysis(rm)
                setattr(R, "ca_row_coords", row_coords)
                setattr(R, "ca_col_coords", col_coords)
                setattr(R, "ca_explained_inertia", explained_inertia)
            if stat == "chi2":
                residuals_df, sorted_residuals, expected_df, chi2_stat, dof = utilsbib.extract_sorted_residual_pairs(rm)
                setattr(R, "chi2_residuals_df", residuals_df)
                setattr(R, "chi2_sorted_residuals", sorted_residuals)
                setattr(R, "chi2_expected_df", expected_df)
                setattr(R, "chi2_chi2_stat", chi2_stat)
                setattr(R, "chi2_dof", dof)           
            if stat == "svd":
                row_projection, singular_values, explained_variance = utilsbib.compute_svd_statistics(rm)
                setattr(R, "svd_row_projection", row_projection)
                setattr(R, "svd_singular_values", singular_values)
                setattr(R, "svd_explained_variance", explained_variance)
            if stat == "log-ratio":
                log_ratio_df, expected_df, sorted_log_ratios = utilsbib.compute_log_ratio(rm)
                setattr(R, "log_ratio_df", log_ratio_df)
                setattr(R, "log_ratio_expected_df", expected_df)
                setattr(R, "log_ratio_sorted_log_ratios", sorted_log_ratios)

        self.relations[concept1][concept2] = R
        
        if to_self:
            R.link_to_self(self)

    def cluster_items(self, items, cluster_by, **kwargs):
        self.relate_concepts(items, cluster_by, include_stats=["cluster", "bicluster"],
                             clean_zeros=True, to_self=True, **kwargs)
        


    # topic modelling
    
    def get_topics(self, v="Processed Abstract", model_type="LDA", n_topics=None, max_topics=10,  max_features=5000, stop_words="english"):
        self.df, self.topic_df = utilsbib.topic_modeling(self.df, v,  model_type=model_type, n_topics=n_topics, max_topics=max_topics,  max_features=max_features, stop_words=stop_words)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.topic_df, f_name=self.res_folder + "\\tables\\topics.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)

    # semantic interdisciplinarity
    
    def get_semantic_interdisciplinarity(self, vrs=[("Processed Abstract", "text")]):
        for v, m in vrs:
            self.df[f"Semantic id {v}"] = self.df[v].apply(lambda x: utilsbib.semantic_interdisciplinarity(x, mode=m))


    # sentiment analysis
    
    def get_sentiment(self, v="Processed Abstract", sentiment_threshold=0.05, top_words=10, correlate=["Year", "Cited by"]):
        self.df, self.sentiment_stats_df = utilsbib.analyze_sentiment(self.df, v, sentiment_threshold=sentiment_threshold, top_words=top_words)
        if self.res_folder is not None:
            utilsbib.to_excel_fancy(self.sentiment_stats_df, f_name=self.res_folder + "\\tables\\sentiment analysis stats.xlsx", autofit=self.autofit, conditional_formatting=self.cond_formatting)
        if len(correlate) > 0:
            self.sentiment_correlations = self.df[["Sentiment Score"] + correlate].corr()

            
    # LLM
    
    def llm_describe_df(self, df_att, **kwargs):
        desc = utilsbib.llm_describe_table(df_att, **kwargs)
        setattr(self, df_att+"_desc", desc)

    def llm_describe_dfs(self, dfs=None, **kwargs):
        if dfs is None:
            dfs = [df for df in dir(self) if df.endswith("_df") if df not in ["show_df"]]
        for df_att in dfs:
            self.llm_describe_df(df_att, **kwargs)

    
    # clustering of the documents
    
    def cluster_documents(self, text_field: str = "Abstract",
                          method: str = "kmeans",
                          n_clusters: int | None = None,
                          k_range: range = range(2, 11),
                          coupling_fields: list[str] | str | None = None,
                          **vectorize_kwargs):
        
        self.df, self.doc_clusters_df, self.new_column = utilsbib.cluster_documents(self.df, text_field=text_field,
                                   method=method, n_clusters=n_clusters,
                                   k_range=k_range, coupling_fields=coupling_fields,
                                   **vectorize_kwargs)


    # REPORTS
    
    def save_report_to_excel(self, output_path: str = "results\\reports\\bibliometric_report.xlsx", **kwargs):
        reportbib.save_excel_report_from_template(self, output_path=output_path, **kwargs)
    
    def save_reports(self, formats=["docx", "xlsx", "pptx", "tex"], f_name=None): # tole ime si obljubil za članek ISSI
        pass


    def save_to_file(self, file_name="biblio.pkl", exclude_dataset=True):
        with open(self.res_folder + "\\" + file_name, "wb") as file:
            import pickle
            if exclude_dataset:
                df0 = self.df
                self.df = None
            pickle.dump(self,  file)
            self.set_data(df0)
        print(f"Analysis saved to {file_name}")
        
    def set_data(self, df):
        self.df = df
        
    def show_data(self, sample=True, n=20):
        utilsbib.make_folders([self.res_folder + "\\sample data"])
        f_name = self.res_folder + "\\sample data\\working data.xlsx"
        if sample:
            df = self.df.sample(n=n)
            f_name = self.res_folder + "\\sample data\\working data sample.xlsx"
        else:
            df = self.df
            f_name = self.res_folder + "\\sample data\\working data.xlsx"
        df.to_excel(f_name, index=False)
        os.startfile(f_name)
        
    def show_df(self, att):
        utilsbib.make_folders([self.res_folder + "\\tmp"])
        f_name = self.res_folder + f"\\tmp\\{att}.xlsx"
        df = getattr(self, att)
        df.to_excel(f_name, index=False)
        os.startfile(f_name)            
        