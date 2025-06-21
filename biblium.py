# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:54:37 2025

@author: Lan.Umek
"""

from bibstats import BiblioStats
from bibplot import BiblioPlot, BiblioGroupPlot
from bibgroup import BiblioGroup
from bibclass import BiblioGroupClassifier

class BiblioAnalysis(BiblioPlot, BiblioGroup):
    
    def compute_all(self):
        pass
    
class BiblioGroupAnalysis(BiblioGroupClassifier, BiblioGroupPlot, BiblioGroup):
    
    def compute_all(self):
        pass