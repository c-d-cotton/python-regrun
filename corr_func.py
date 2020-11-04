#!/usr/bin/env python3
"""
OLD - need to rewrite if want to use
Maybe delete
"""
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

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
            df = importattr(__projectdir__ / Path('regrun_func.py'), 'extenddf')(df, variables)

        variables = df[variables].dropna().values.transpose()
        
    corrcoef = np.corrcoef(variables)
    if printoutput is True:
        print(corrcoef)

    tabular = importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'matrixtotabularfull')(corrcoef.tolist(), decimalpoints = 3, rowindex = variablenames, colindex = [''] + variablenames, colalign = 'l|' + 'c' * len(variablenames), hlines = [1], savenames = savenames, appendnames = appendnames, colindexstandard2turn = colindexstandard2turn)


    return(tabular)


