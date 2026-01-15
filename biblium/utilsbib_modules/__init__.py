# -*- coding: utf-8 -*-
"""
Utilsbib Modules - Modular utilities for bibliometric analysis.

This package organizes the utilsbib functions into logical modules:

- core: Basic utilities, progress bars, file operations
- countries: Country mappings and extraction
- excel_io: Excel reading/writing with formatting
- counting: Entity counting, occurrence analysis
- text: Text processing, keywords, stopwords
- stats: Bibliometric indices and laws
- network: Graph analysis, community detection
- groups: Group-based analysis

Note: For backward compatibility, all functions are still accessible from
the main utilsbib module. This package provides a more organized alternative.
"""

# Core utilities
from biblium.utilsbib_modules.core import (
    make_folder,
    make_folders,
    progress_bar,
    progress_apply,
    rename_attributes,
    first_existing,
    build_exclusion_regex,
    build_inclusion_regex,
    reconstruct_mapping,
    reconstruct_mapping_from_excel,
    fd,
    TQDM_AVAILABLE,
)

# Country utilities
from biblium.utilsbib_modules.countries import (
    df_countries,
    domain_dct,
    c_off_dct,
    code_dct,
    code_dct_r,
    country_iso3_dct,
    continent_dct,
    code_to_coords,
    l_countries,
    eu_countries,
    correct_country_name,
    split_ca,
    parse_mail,
    get_ca_country_scopus,
    get_ca_country_wos,
    get_ca_country,
    add_ca_country_df,
    extract_countries_from_affiliations,
    openalex_map_country_codes,
    openalex_add_corresponding_country,
)

# Excel I/O
from biblium.utilsbib_modules.excel_io import (
    to_excel_fancy,
    save_descriptives_to_excel,
    save_plot,
)

# Counting utilities
from biblium.utilsbib_modules.counting import (
    count_occurrences,
    match_items_and_compute_binary_indicators,
    get_entity_stats,
)

# Text processing
from biblium.utilsbib_modules.text import (
    stopwords_file,
    ensure_wordnet,
    preprocess_keywords,
    merge_keywords_columns,
    merge_text_columns,
    build_combined_text,
    process_text_column,
)

# Statistics
from biblium.utilsbib_modules.stats import (
    # Core h-index variants
    h_index,
    g_index,
    hg_index,
    c_index,
    tapered_h_index,
    chi_index,
    a_index,
    r_index,
    h2_index,
    w_index,
    gini_index,
    # New h-index variants (2.2.6)
    m_index,
    ar_index,
    m_quotient,
    h2_upper_index,
    e_index,
    q2_index,
    hw_index,
    pi_index,
    i10_index,
    i100_index,
    v_index,
    h_norm_index,
    h_frac_index,
    profit_index,
    compute_all_h_indices,
    # Bibliometric laws
    compute_lotka_distribution,
    evaluate_lotka_fit,
    compute_bradford_distribution,
    compute_zipf_distribution_from_counts,
    evaluate_zipf_fit,
    evaluate_prices_law,
    evaluate_pareto_principle,
    # Citation distribution
    analyze_citation_distribution,
    # Collaboration analysis
    analyze_collaboration,
    # Sentiment analysis
    analyze_sentiment_advanced,
    get_sentiment_by_entity,
    compare_sentiment_groups,
    SCIENTIFIC_CERTAINTY_MARKERS,
    SCIENTIFIC_POSITIVE_WORDS,
    SCIENTIFIC_NEGATIVE_WORDS,
)

# Network analysis
from biblium.utilsbib_modules.network import (
    build_cooccurrence_matrix,
    matrix_to_network,
    normalize_symmetric_matrix,
    louvain_partition,
    greedy_modularity_partition,
    label_propagation_partition,
    add_partitions,
    compute_basic_stats,
    compute_centralities,
    nodes_to_dataframe,
    save_network,
    save_to_pajek,
)

# Group analysis
from biblium.utilsbib_modules.groups import (
    generate_group_matrix,
    merge_group_performances,
    count_occurrences_across_groups,
    compute_group_intersections,
    compare_continuous_by_binary_groups,
    group_entity_stats,
)

# Caching utilities
from biblium.utilsbib_modules.caching import (
    CacheKey,
    ResultCache,
    InstanceCache,
    get_cache,
    set_cache_enabled,
    clear_cache,
    get_cache_stats,
    cached,
    cached_property,
    cache_cooccurrence_key,
    cache_network_key,
)

# Data quality
from biblium.utilsbib_modules.data_quality import (
    AuthorCluster,
    normalize_author_name,
    parse_author_name,
    compute_author_similarity,
    disambiguate_authors,
    apply_author_disambiguation,
    get_author_variants_report,
    detect_duplicate_documents,
    remove_duplicates,
    analyze_missing_data,
    get_data_quality_score,
)

__all__ = [
    # Core
    "make_folder", "make_folders", "progress_bar", "progress_apply",
    "rename_attributes", "first_existing", "build_exclusion_regex",
    "build_inclusion_regex", "reconstruct_mapping", "reconstruct_mapping_from_excel",
    "fd", "TQDM_AVAILABLE",
    # Countries
    "df_countries", "domain_dct", "c_off_dct", "code_dct", "code_dct_r",
    "country_iso3_dct", "continent_dct", "code_to_coords", "l_countries",
    "eu_countries", "correct_country_name", "split_ca", "parse_mail",
    "get_ca_country_scopus", "get_ca_country_wos", "get_ca_country",
    "add_ca_country_df", "extract_countries_from_affiliations",
    "openalex_map_country_codes", "openalex_add_corresponding_country",
    # Excel I/O
    "to_excel_fancy", "save_descriptives_to_excel", "save_plot",
    # Counting
    "count_occurrences", "match_items_and_compute_binary_indicators", "get_entity_stats",
    # Text
    "stopwords_file", "ensure_wordnet", "preprocess_keywords",
    "merge_keywords_columns", "merge_text_columns", "build_combined_text",
    "process_text_column",
    # Stats - h-index variants
    "h_index", "g_index", "hg_index", "c_index", "tapered_h_index",
    "chi_index", "a_index", "r_index", "h2_index", "w_index", "gini_index",
    "m_index", "ar_index", "m_quotient", "h2_upper_index", "e_index",
    "q2_index", "hw_index", "pi_index", "i10_index", "i100_index",
    "v_index", "h_norm_index", "h_frac_index", "profit_index",
    "compute_all_h_indices",
    # Stats - bibliometric laws
    "compute_lotka_distribution", "evaluate_lotka_fit", "compute_bradford_distribution",
    "compute_zipf_distribution_from_counts", "evaluate_zipf_fit",
    "evaluate_prices_law", "evaluate_pareto_principle",
    "analyze_citation_distribution", "analyze_collaboration",
    # Network
    "build_cooccurrence_matrix", "matrix_to_network", "normalize_symmetric_matrix",
    "louvain_partition", "greedy_modularity_partition", "label_propagation_partition",
    "add_partitions", "compute_basic_stats", "compute_centralities",
    "nodes_to_dataframe", "save_network", "save_to_pajek",
    # Groups
    "generate_group_matrix", "merge_group_performances", "count_occurrences_across_groups",
    "compute_group_intersections", "compare_continuous_by_binary_groups", "group_entity_stats",
    # Caching
    "CacheKey", "ResultCache", "InstanceCache", "get_cache", "set_cache_enabled",
    "clear_cache", "get_cache_stats", "cached", "cached_property",
    "cache_cooccurrence_key", "cache_network_key",
    # Data quality
    "AuthorCluster", "normalize_author_name", "parse_author_name",
    "compute_author_similarity", "disambiguate_authors", "apply_author_disambiguation",
    "get_author_variants_report", "detect_duplicate_documents", "remove_duplicates",
    "analyze_missing_data", "get_data_quality_score",
    # Preprocessing
    "preprocess_pubmed", "preprocess_lens", "preprocess_dimensions",
    "preprocess_bibliographic_data", "detect_database", "standardize_columns",
    # Citations
    "compute_field_normalized_citations", "compute_citation_classes",
    "get_citation_summary", "detect_self_citations", "get_self_citation_summary",
    "extract_dois_from_references", "match_references_by_doi",
    "build_citation_network_from_dois", "compute_citation_velocity",
    # Topic Modeling
    "topic_modeling", "get_topic_summary", "compute_topic_coherence",
    # Time Series
    "compute_publication_trend", "compute_citation_evolution",
    "compute_growth_rates", "predict_future_publications",
    "compute_doubling_time", "compute_half_life", "detect_trend_changes",
    "fit_growth_model", "fit_life_cycle_model",
]

# Preprocessing
from biblium.utilsbib_modules.preprocessing import (
    preprocess_pubmed,
    preprocess_lens,
    preprocess_dimensions,
    preprocess_bibliographic_data,
    detect_database,
    standardize_columns,
)

# Citations
from biblium.utilsbib_modules.citations import (
    compute_field_normalized_citations,
    compute_citation_classes,
    get_citation_summary,
    detect_self_citations,
    get_self_citation_summary,
    extract_dois_from_references,
    match_references_by_doi,
    build_citation_network_from_dois,
    compute_citation_velocity,
)

# Topic Modeling
from biblium.utilsbib_modules.topic_modeling import (
    topic_modeling,
    get_topic_summary,
    compute_topic_coherence,
)

# Time Series
from biblium.utilsbib_modules.time_series import (
    compute_publication_trend,
    compute_citation_evolution,
    compute_growth_rates,
    predict_future_publications,
    compute_doubling_time,
    compute_half_life,
    detect_trend_changes,
    fit_growth_model,
    fit_life_cycle_model,
)
