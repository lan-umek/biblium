# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:29:43 2025

@author: Lan.Umek
"""

from bibplot import BiblioPlot, BiblioGroupPlot


import pandas as pd

class BaseGroupAnalysis(BiblioGroupPlot):
    """
    Generic analysis class for different SDG dimensions.

    Subclasses must define:
        group_desc (str): Description of the group/dimension.
        res_folder (str): Name of the results folder.
    """
    group_desc = None
    res_folder = None

    def __init__(self, f_name=None, db="", df: pd.DataFrame=None, **kwargs):
        """
        Initialize the analysis group with specified group description and results folder.

        Args:
            f_name (str, optional): File name or identifier.
            db (str, optional): Database connection string or path.
            df (pandas.DataFrame, optional): DataFrame containing documents and metadata.
            **kwargs: Additional parameters passed to BiblioGroupPlot.
        """
        # Enforce the class-level results folder
        kwargs.pop("res_folder", None)

        super().__init__(
            f_name=f_name,
            db=db,
            df=df,
            group_desc=self.group_desc,
            res_folder=self.res_folder,
            **kwargs,
        )


class ScienceGroupAnalysis(BaseGroupAnalysis):
    """
    Science dimension analysis class.

    Uses group_desc = "Science Dimension" and res_folder = "results-dimension".
    """
    group_desc = "Science Dimension"
    res_folder = "results-dimension"

class AreaGroupAnalysis(BaseGroupAnalysis):
    group_desc = "Area"
    res_folder = "results-area"

class FiledGroupAnalysis(BaseGroupAnalysis):
    group_desc = "Field"
    res_folder = "results-field"

class SDGGroupAnalysisGoals(BaseGroupAnalysis):
    """
    SDG goals analysis class.

    Dynamically computes `group_desc` based on any DataFrame columns containing "SDG "
    and uses res_folder = "results-goals".
    """
    res_folder = "results-goals"

    def __init__(self, f_name=None, db="", df: pd.DataFrame=None, **kwargs):
        """
        Initialize the SDG goals analysis group with dynamic description.

        Args:
            f_name (str, optional): File name or identifier.
            db (str, optional): Database connection string or path.
            df (pandas.DataFrame, optional): DataFrame containing documents and metadata.
            **kwargs: Additional parameters passed to BiblioGroupPlot.
        """
        # Compute dynamic description based on DataFrame columns
        if df is not None:
            self.group_desc = df[[c for c in df.columns if "SDG " in c]]
        else:
            self.group_desc = []

        # Call the base initializer with dynamic group_desc
        super().__init__(
            f_name=f_name,
            db=db,
            df=df,
            **kwargs,
        )



class SDGGroupAnalysisPerspective(BaseGroupAnalysis):
    res_folder = "results-perspectives"

    def __init__(self, f_name=None, db="", df: pd.DataFrame=None, **kwargs):
        if df is not None:
            self.group_desc = df[[c for c in df.columns if "perspective" in c]]
        else:
            self.group_desc = []

        # Call the base initializer with dynamic group_desc
        super().__init__(
            f_name=f_name,
            db=db,
            df=df,
            **kwargs,
        )

class SDGGroupAnalysisDimension(BaseGroupAnalysis):
    res_folder = "results-dimensions"

    def __init__(self, f_name=None, db="", df: pd.DataFrame=None, **kwargs):
        if df is not None:
            self.group_desc = df[[c for c in df.columns if "dimension" in c]]
        else:
            self.group_desc = []

        # Call the base initializer with dynamic group_desc
        super().__init__(
            f_name=f_name,
            db=db,
            df=df,
            **kwargs,
        )