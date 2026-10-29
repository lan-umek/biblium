# -*- coding: utf-8 -*-
"""
Paneli za novosti 2.19 (vsi temeljijo na ``SimpleAnalysisPanel``).

Registracija: ``register_v219_panels(workspace)`` v ``gui/app.py``;
postavke stranskega menija: ``V219_SIDEBAR_SECTIONS`` v ``gui/components/sidebar.py``.
"""
from __future__ import annotations

from typing import Dict, Type

from biblium.gui.panels.v219.indicators_panels import (
    CSSPanel, PercentileIndicatorsPanel, ConcentrationPanel, UncitednessPanel,
    BootstrapCIPanel, AuthorCareersPanel, HeapsLawPanel, ReferenceAgePanel,
    CitedSourcesPanel,
)
from biblium.gui.panels.v219.quality_panels import (
    PrismaPanel, MetadataCompletionPanel, JournalRankingPanel, FundingPanel,
)
from biblium.gui.panels.v219.reference_panels import (
    RPYSPatternsPanel, DualMapPanel, SnowballPanel,
)
from biblium.gui.panels.v219.dynamics_panels import (
    DiachronicNetworkPanel, ClusterTimelinePanel, StructuralVariationPanel,
    FuzzyClustersPanel, TermExtractionPanel,
)

V219_PANELS: Dict[str, Type] = {
    # indicators
    "css": CSSPanel,
    "percentile_indicators": PercentileIndicatorsPanel,
    "concentration": ConcentrationPanel,
    "uncitedness": UncitednessPanel,
    "bootstrap_ci": BootstrapCIPanel,
    "author_careers": AuthorCareersPanel,
    "heaps_law": HeapsLawPanel,
    # references
    "reference_age": ReferenceAgePanel,
    "cited_sources": CitedSourcesPanel,
    "rpys_patterns": RPYSPatternsPanel,
    "dual_map": DualMapPanel,
    "snowball": SnowballPanel,
    # data quality / review
    "prisma": PrismaPanel,
    "metadata_completion": MetadataCompletionPanel,
    "journal_ranking": JournalRankingPanel,
    "funding": FundingPanel,
    # dynamics / text
    "diachronic_network": DiachronicNetworkPanel,
    "cluster_timeline": ClusterTimelinePanel,
    "structural_variation": StructuralVariationPanel,
    "fuzzy_clusters": FuzzyClustersPanel,
    "term_extraction": TermExtractionPanel,
}

V219_SIDEBAR_SECTIONS = [
    {
        "title": "INDICATORS",
        "items": [
            {"id": "css", "text": "CSS Citation Classes", "icon": "🏅"},
            {"id": "percentile_indicators", "text": "Percentile Indicators", "icon": "📊"},
            {"id": "concentration", "text": "Concentration (Lorenz)", "icon": "📐"},
            {"id": "uncitedness", "text": "Uncitedness", "icon": "🫥"},
            {"id": "bootstrap_ci", "text": "Bootstrap CI", "icon": "🎲"},
            {"id": "author_careers", "text": "Author Careers", "icon": "🧑‍🔬"},
            {"id": "heaps_law", "text": "Heaps' Law", "icon": "📈"},
            {"id": "funding", "text": "Funding Analysis", "icon": "💶"},
        ],
    },
    {
        "title": "REFERENCES",
        "items": [
            {"id": "reference_age", "text": "Reference Age / Price", "icon": "⏳"},
            {"id": "cited_sources", "text": "Cited Sources", "icon": "📚"},
            {"id": "rpys_patterns", "text": "RPYS & Patterns", "icon": "🔭"},
            {"id": "dual_map", "text": "Dual-Map Overlay", "icon": "🗺️"},
            {"id": "snowball", "text": "Snowball (OpenAlex)", "icon": "❄️"},
        ],
    },
    {
        "title": "DYNAMICS & TEXT",
        "items": [
            {"id": "diachronic_network", "text": "Diachronic Networks", "icon": "🎞️"},
            {"id": "cluster_timeline", "text": "Cluster Timeline", "icon": "📅"},
            {"id": "structural_variation", "text": "Structural Variation", "icon": "🧬"},
            {"id": "fuzzy_clusters", "text": "Fuzzy Doc Clusters", "icon": "🌫️"},
            {"id": "term_extraction", "text": "Term Extraction", "icon": "🔤"},
        ],
    },
    {
        "title": "REVIEW & QUALITY",
        "items": [
            {"id": "prisma", "text": "PRISMA Diagram", "icon": "🧾"},
            {"id": "metadata_completion", "text": "Metadata Completion", "icon": "🩹"},
            {"id": "journal_ranking", "text": "Journal Rankings", "icon": "🏆"},
        ],
    },
]


def register_v219_panels(workspace) -> None:
    for pid, cls in V219_PANELS.items():
        workspace.register_panel(pid, cls)


__all__ = ["V219_PANELS", "V219_SIDEBAR_SECTIONS", "register_v219_panels"]
