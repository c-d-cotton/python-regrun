#!/usr/bin/env python3
"""
OLD - need to rewrite if want to use
Maybe delete
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

def corrtabular(variables, df = None, variablenames = None, printoutput = False, savenames = None, appendnames = None, colindexstandard2turn = False, extenddf = True, copydf = True):
    """
    Get tabular version of correlation matrix.

    If specify df then variables is a list of names. Then do not need to specify variablenames.
    Otherwise, do need to specify variablenames.

    extenddf = True means I can use L. etc. and copydf means I copy the df before doing so.
    """
    import copy
    import numpy as np

    if df is not None:
        if variablenames is None:
            variablenames = variables

        if extenddf is True:
            if copydf is True:
                df = copy.deepcopy(df)
            from regrun_func import extenddf
            df = extenddf(df, variables)

        variables = df[variables].dropna().values.transpose()
        
    corrcoef = np.corrcoef(variables)
    if printoutput is True:
        print(corrcoef)

    sys.path.append(str(__projectdir__ / Path('submodules/python-texoutput/')))
    from regoutput_func import matrixtotabularfull
    tabular = matrixtotabularfull(corrcoef.tolist(), decimalpoints = 3, rowindex = variablenames, colindex = [''] + variablenames, colalign = 'l|' + 'c' * len(variablenames), hlines = [1], savenames = savenames, appendnames = appendnames, colindexstandard2turn = colindexstandard2turn)


    return(tabular)


