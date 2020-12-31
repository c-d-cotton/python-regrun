#!/usr/bin/env python3
"""
Basic functions for running regressions and converting them into tables
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Definitions:{{{1
coeffdecimal_default = 3


# Construct Data:{{{1
def extenddf(df, variables):
    """
    Add variables with prefixes to df. The prefixes are:
    D(2). (difference)
    L(1). (lag)
    F(3). (forward)
    log. (log)
    exp. (exponential)

    Can use these in succession i.e. D1.F.x.
    """
    import collections
    import numpy as np
    import re

    # DECIDED NOT DO THIS BECAUSE THEN NEED TO REPLACE D. in names as well.
    # first replace D., L. and F. with D1., L1. and F1.
    # for i in range(len(variables)):
    #     for x, y in [('D', 'D1'), ('L', 'L1'), ('F', 'F1')]:
    #         variables[i] = variables[i].replace('.' + x + '.', '.' + y + '.')
    #         if variables[i].startswith(x + '.'):
    #             variables[i] = variables[i].replace(x, y, 1)

    # need regex to match for suffix terms
    compiled = re.compile('^(D[0-9]*|F[0-9]*|L[0-9]*|log|exp)\.')

    # needs to be an ordered dict since add variables without suffixes first
    extendvardict = collections.OrderedDict()
    for var in variables:
        # get the prefix of all bits at the start.
        suffix = ''
        while True:
            match = compiled.search(var)
            if match is not None:
                suffix = suffix + match.group(0)
                var = var[len(match.group(0)): ]
            else:
                break

        # add to extendvardict
        if suffix != '':
            
            while '.' in suffix:
                # if suffix = 'D1.L2. then suffix2 = L2 and suffix = D1.
                suffixsplit = suffix.split('.')
                suffix2 = suffixsplit[-2]
                del suffixsplit[-2]
                suffix = '.'.join(suffixsplit)

                if var not in extendvardict:
                    extendvardict[var] = []
                if suffix2 not in extendvardict[var]:
                    extendvardict[var].append(suffix2)

                # adjust var so that add the suffix2 to front so if var = 'x' and suffix2 = 'L2' then var = 'L2.x'
                var = suffix2 + '.' + var
    
    # end number regex
    endnumre = re.compile('[0-9]+')
    endnumre = re.compile('(.*)([0-9])+')
    # add all additional variables to the dataframe if they do not exist:
    for var in extendvardict:
        for suffix in extendvardict[var]:
            # skip if already exists
            if suffix + var in df.columns:
                continue

            # get number at end of suffix (relevant for L, D, F)
            match = endnumre.match(suffix)
            if match is not None:
                letter = match.group(1)
                num = match.group(2)
            else:
                letter = suffix
                num = None

            if letter == 'D':
                if num is None:
                    num = '1'
                df[suffix + '.' + var] = df[var].diff(int(num))
            elif letter == 'L':
                if num is None:
                    num = '1'
                df[suffix + '.' + var] = df[var].shift(int(num))
            elif letter == 'F':
                if num is None:
                    num = '1'
                df[suffix + '.' + var] = df[var].shift(-int(num))
            elif letter == 'log':
                df[suffix + '.' + var] = np.log(df[var])
            elif letter == 'exp':
                df[suffix + '.' + var] = np.exp(df[var])

    return(df)


def extenddf_test():
    import pandas as pd

    df = pd.DataFrame()
    df['x'] = [1,2,3]
    df['y'] = [4,5,6]
    df['z'] = [7,8,9]
    df['hello.z'] = [10, 11, 12]

    df = extenddf(df, ['L.x', 'F2.y', 'D.L1.z', 'log.z', 'hello.z'])

    print(df)


# OLS:{{{1
def genolsmatrices(yvar, Xvar, df, addconstant = True, Xappend = False, extenddf = True, copydf = True, dropmissing = True):
    """
    If df is None then Yvariables and Xvariables are matrices. Otherwise, they are names.
    Yvariables can be list or just one. If just one, use this for all regressions.

    addconstant = True means add a constant to the regression. addconstant can also be specified as a list which varies by regression

    If Xappend = True then add the matrices for X together so if I have Xvariables = [['x1'], ['x2']] then actually use Xvariables = [['x1'], ['x1', 'x2']].
    Note that for Xappend to work then I need the variables in each regression to be the same length.

    extenddf = True means that I allow for L., F. etc. when using df.
    copydf means I make a copy of the dataframe. Default: True. It only really makes sense not to make a copy if I'm using the same dataframe for each regression or if I don't care about maintaining the same df

    dropmissing means I drop the missing values from the matrices. Default: True.
    """

    # convert yvar, Xvar, df into lists:{{{
    lenlist = None
    if isinstance(yvar, list):
        yvarlist = True
        lenlist = len(yvar)
    else:
        yvarlist = False
    if isinstance(Xvar[0], list):
        Xvarlist = True
        if lenlist is None:
            lenlist = len(Xvar)
        else:
            if len(Xvar) != lenlist:
                raise ValueError('Xvarlist not the same length as lenlist')
    else:
        Xvarlist = False
    if isinstance(df, list):
        dflist = True
        if lenlist is None:
            lenlist = len(df)
        else:
            if len(df) != lenlist:
                raise ValueError('dflist not the same length as lenlist')
    else:
        dflist = False

    if lenlist is None:
        raise ValueError('At least one of yvar, Xvar or df should be a list')

    if yvarlist is False:
        yvar = [yvar] * lenlist
    if Xvarlist is False:
        Xvar = [Xvar] * lenlist
    if dflist is False:
        df = [df] * lenlist
    # convert yvar, Xvar, df into lists:}}}

    if not isinstance(addconstant, list):
        addconstant = [addconstant] * len(yvar)

    if extenddf is True:
        for i in range(len(df)):
            # take a copy of the dataframe to avoid 
            if copydf is True:
                df[i] = copy.deepcopy(df[i])
            df[i] = extenddf(df[i], Xvar[i])
            
        
    ymatrices = []
    Xmatrices = []
    for i in range(0, len(yvar)):
        # get y, X
        ymatrices.append( df[i][yvar[i]] )
        Xmatrices.append( df[i][Xvar[i]] )

        if addconstant[i] is True:
            Xmatrices[i] = sm.add_constant(Xmatrices[i])


    return(ymatrices, Xmatrices)


def runols(ymatrices, Xmatrices, regruntype = None, regoptions = None, fitoptions = None):
    """
    Run OLS.
    Incorporate save and print function directly here.
    """

    if regruntype is None:
        regruntype = sm.OLS
    if not isinstance(regruntype, list):
        regruntype = [regruntype] * len(ymatrices)

    if regoptions is None:
        regoptions = {'missing': 'drop'}
    if not isinstance(regoptions, list):
        regoptions = [regoptions] * len(ymatrices)

    if fitoptions is None:
        fitoptions = {}
    if not isinstance(fitoptions, list):
        fitoptions = [fitoptions] * len(ymatrices)

    modelfitlist = []
    for i in range(len(Xmatrices)):
        result = regruntype[i](ymatrices[i], Xmatrices[i], **regoptions[i]).fit(**fitoptions[i])
        modelfitlist.append(result)

    return(modelfitlist)


# Print/save regressions:{{{1
def saveregs(modelfitlist, printsummary = False, save = None, savetxt = None, savetex = None, append = None, appendtxt = None, appendtex = None):
    """
    modelfitlist is a list of statsmodels modelfitlist objects.

    if element in save then put the name without the extension in savetex and savetxt. Same for append.
    """
    if save is None:
        save = []
    if savetxt is None:
        savetxt = []
    if savetex is None:
        savetex = []
    if append is None:
        append = []
    if appendtxt is None:
        appendtxt = []
    if appendtex is None:
        appendtex = []
    
    if not isinstance(save, list):
        save = [save]
    if not isinstance(savetxt, list):
        savetxt = [savetxt]
    if not isinstance(savetex, list):
        savetex = [savetex]
    if not isinstance(append, list):
        append = [append]
    if not isinstance(appendtxt, list):
        appendtxt = [appendtxt]
    if not isinstance(appendtex, list):
        appendtex = [appendtex]
    
    for i in range(len(save)):
        # remove extension
        save[i] = save[i][: -len(os.path.splitext(save[i])[1])]
        savetxt.append(save[i] + '.txt')
        savetex.append(save[i] + '.tex')
    for i in range(len(append)):
        # remove extension
        append[i] = append[i][: -len(os.path.extension(append[i]))]
        appendtxt.append(append[i] + '.txt')
        appendtex.append(append[i] + '.tex')

    summarytxt = []
    for i in range(len(modelfitlist)):
        # this doens't work when using robust standard errors
        try:
            summarytxt.append(modelfitlist[i].summary().as_text())
        except Exception:
            None
    alltxt = '\n\n'.join(summarytxt)

    if printsummary is True:
        print(alltxt)
        
    summarytex = []
    for i in range(len(modelfitlist)):
        # this doesn't work when using robust standard errors
        try:
            summarytex.append(modelfitlist[i].summary().as_latex())
        except Exception:
            None
    alltex = '\n\n'.join(summarytex)

    # save filenames
    for i in range(len(savetxt)):
        with open(savetxt[i], 'w+') as f:
            f.write(alltxt)
    for i in range(len(savetex)):
        with open(savetex[i], 'w+') as f:
            f.write(alltex)

    for i in range(len(appendtxt)):
        with open(appendtxt[i], 'a+') as f:
            f.write(alltxt + '\n\n')
    for i in range(len(appendtex)):
        with open(appendtex[i], 'a+') as f:
            f.write(alltex + '\n\n')


def testdf():
    np.random.seed(1)

    data = np.random.normal(size = [100, 4])
    df = pd.DataFrame(data, columns = ['x1', 'x2', 'e1', 'e2'])

    df['y1'] = df['x1'] + df['e1']
    df['y2'] = df['x2'] + df['e2']

    return(df)


def runols_test_ysame_dfsame():
    df = testdf()

    ymatrices, Xmatrices = genolsmatrices('y1', [['x1'], ['x2'], ['x1', 'x2']], df)

    modelfitlist = runols(ymatrices, Xmatrices)

    # this isn't actually doing something
    # need to do printsummary = True of add save options if want to do something
    saveregs(modelfitlist, printsummary = False)

    return(modelfitlist)


def runols_test_alldiff():
    df = testdf()

    dffirsthalf = df[df.index < 0.5 * len(df)]

    ymatrices, Xmatrices = genolsmatrices(['y1', 'y1', 'y2', 'y1'], [['x1'], ['x1', 'x2'], ['x1'], ['x1']], [df, df, df, dffirsthalf])

    modelfitlist = runols(ymatrices, Xmatrices)

    saveregs(modelfitlist, printsummary = True)

    return(modelfitlist)


# Tabular - one column is one regression:{{{1
def getsamecoeffmatrices(modelfitlist, coeffregnames = None):
    """
    Get details on the betas, standard deviations and pvalues of a set of coefficients from a list of regressions
    """

    # get the list of variables in the regression if not specified
    if coeffregnames is None:
        coeffregnames = []
        for result in modelfitlist:
            thecoeff = list(result.params.index)
            for coeff in thecoeff:
                if coeff not in coeffregnames:
                    coeffregnames.append(coeff)

    betamatrix = [[None] * len(modelfitlist) for i in range(len(coeffregnames))]
    pvalmatrix = [[None] * len(modelfitlist) for i in range(len(coeffregnames))]
    sematrix = [[None] * len(modelfitlist) for i in range(len(coeffregnames))]
    
    for coeffi in range(len(coeffregnames)):
        for regj in range(len(modelfitlist)):
            if coeffregnames[coeffi] in modelfitlist[regj].params:
                betamatrix[coeffi][regj] = decimal.Decimal(modelfitlist[regj].params[coeffregnames[coeffi]])
                pvalmatrix[coeffi][regj] = decimal.Decimal(modelfitlist[regj].pvalues[coeffregnames[coeffi]])
                sematrix[coeffi][regj] = decimal.Decimal(modelfitlist[regj].bse[coeffregnames[coeffi]])

    return(coeffregnames, betamatrix, pvalmatrix, sematrix)


def getsamecoeffmatrices_test():
    modelfitlist = runols_test_ysame_dfsame()

    coeffregnames, betamatrix, pvalmatrix, sematrix = getsamecoeffmatrices(modelfitlist)
    print(coeffregnames)
    print(betamatrix)


def getparammatrix(modelfitlist, paramlist):
    """
    Get a list of parameters from a list of model.fit() from statsmodels

    List of potential parameters:
    http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html
    """
    parammatrix = [[None] * len(modelfitlist) for i in range(len(paramlist))]


    for i in range(len(modelfitlist)):
        for j in range(len(paramlist)):
            parammatrix[j][i] = eval('modelfitlist[i].' + paramlist[j])

    return(parammatrix)


def getparammatrix_test():
    modelfitlist = runols_test_ysame_dfsame()

    parammatrix = getparammatrix(modelfitlist, ['nobs'])
    print(parammatrix)


def getmultiregtabular_sm(
    # key arguments
    modelfitlist, ytabnames,
    # optional arguments for parsing the regressions
    xregnames = None, xtabnames = None, paramregnames = 'def', paramtabnames = None,
    # optional arguments for individual components of the table
    stardict = 'def', coeffdecimal = None, paramdecimal = None,
    # optional printing arguments
    printlist = True, printmaxcolsize = 'def',
    # optional arguments around what return
    returnmatrices = False, returntabsecs = False, returntabularbody = False, 
    # misc arguments
    savename = None,
    ):
    """
    This returns a tabular for multiple regressions where each column represents a regression
    Key inputs: modelfitlist i.e. list of model.fit() and ytabnames
    ytabnames should be in format 'rgdpgr' or ['rgdpgr', 'rgdpgr1', 'rgdpgr2']

    Optional arguments for parsing the regressions:
    xregnames allows me to only select certain coefficients to be in the tabular
    xtabnames allows me to specify the name of these variables in the table if I want them to have different names
    paramregnames allows me to specify which parameters I include in the parameter part of the table and paramtabnames lets me change their name

    The other arguments are the same as in getmultiregtabular
    """

    # this gets the matrices needed for the coefficient part of the tabular
    xregnames, betamatrix, pvalmatrix, sematrix = getsamecoeffmatrices(modelfitlist, coeffregnames = xregnames)
    # specify xtabnames if not specified
    if xtabnames is None:
        xtabnames = xregnames

    # this gets the matrix needed for the parameter part of the tabular
    if paramregnames == 'def':
        paramregnames = ['nobs']
        paramtabnames = ['N']
        paramdecimal = [0]
    parammatrix = getparammatrix(modelfitlist, paramregnames)
    # specify paramtabnames if not specified
    if paramtabnames is None:
        paramtabnames = paramregnames

    sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-general/')))
    from tabular_func import getmultiregtabular
    ret = getmultiregtabular(
    # necessary arguments for creating the basic tables
    ytabnames, xtabnames, paramtabnames, betamatrix, pvalmatrix, sematrix, parammatrix,
    # optional arguments for individual components of the table
    stardict = stardict, coeffdecimal = coeffdecimal, paramdecimal = paramdecimal,
    # optional printing arguments
    printlist = printlist, printmaxcolsize = printmaxcolsize,
    # optional arguments around what return
    returnmatrices = returnmatrices, returntabsecs = returntabsecs, returntabularbody = returntabularbody, 
    # misc arguments
    savename = savename,
    )

    return(ret)


def getmultiregtabular_test_tabular():
    """
    Basic tabular example.
    """
    df = testdf()

    yvar = 'y1'
    Xvar = [['x1'], ['x2'], ['x1', 'x2']]
    ymatrices, Xmatrices = genolsmatrices(yvar, Xvar, df)

    modelfitlist = runols(ymatrices, Xmatrices)

    getmultiregtabular_sm(modelfitlist, ytabnames = yvar, savename = __projectdir__ / Path('temp/regrun/multiregtest_tabular.tex'))


def getmultiregtabular_test_xcoeffreduce():
    """
    Example of only showing some of the X regressors in a regression.
    """
    df = testdf()

    yvar = 'y1'
    Xvar = [['x1'], ['x2'], ['x1', 'x2']]
    ymatrices, Xmatrices = genolsmatrices(yvar, Xvar, df)

    modelfitlist = runols(ymatrices, Xmatrices)

    getmultiregtabular_sm(modelfitlist, ytabnames = yvar, savename = __projectdir__ / Path('temp/regrun/multiregtest_xcoeffreduce.tex'), xregnames = ['x1'])


def getmultiregtabular_test_paramchange():
    """
    Example of using different parameters in the tabular table.
    """
    df = testdf()

    yvar = 'y1'
    Xvar = [['x1'], ['x2'], ['x1', 'x2']]
    ymatrices, Xmatrices = genolsmatrices(yvar, Xvar, df)

    modelfitlist = runols(ymatrices, Xmatrices)

    getmultiregtabular_sm(modelfitlist, ytabnames = yvar, savename = __projectdir__ / Path('temp/regrun/multiregtest_paramchange.tex'), paramregnames = ['nobs', 'rsquared'], paramtabnames = ['N', '$R^2$'], paramdecimal = [0, 3])


def getmultiregtabular_test_tabsecmerge():
    """
    Example of including an additional row in the tabular with details about the dataset.
    """
    df = testdf()

    dffirsthalf = df[df.index < 0.5 * len(df)]

    yvar = ['y1', 'y1', 'y2', 'y1']
    Xvar = [['x1'], ['x1', 'x2'], ['x1'], ['x1']]
    dflist = [df, df, df, dffirsthalf]
    ymatrices, Xmatrices = genolsmatrices(yvar, Xvar, dflist)

    modelfitlist = runols(ymatrices, Xmatrices)

    ynames_tabsec, vcoeff_tabsec, param_tabsec = getmultiregtabular_sm(modelfitlist, yvar, returntabsecs = True, savename = __projectdir__ / Path('temp/regrun/multiregtest_tabsecmerge.tex'))

    # add tabsec specifying which data using
    # first generate basic list of list
    df_lofl = [['Data', 'full', 'full', 'full', 'first half']]
    # then covert to tabsec
    sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-general/')))
    from tabular_func import tabularconvert
    df_tabsec = tabularconvert(df_lofl)

    # merge tabsecs together
    sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-general/')))
    from tabular_func import mergetabsecs
    mergetabsecs([ynames_tabsec, vcoeff_tabsec, param_tabsec, df_tabsec], colalign = 'l' * (len(modelfitlist) + 1), hlines = 'all', savename = __projectdir__ / Path('temp/regrun/multiregtest_tabsecmerge.tex'))


# Tabular - one cell is one regression NEED TO ADJUST:{{{1
def getregmatrices_complex(modelfitlist, coefflist, numcol):
    """
    CURRENTLY UNUSED - NEED TO TIDY UP

    Difference with getsamecoeffmatrices is that I specify individual regressions for each cell rather than column
    And that I specify the independent variable for each cell rather than have them be the same across rows
    Coefficients can be specified by individual cell or by column so either len(coefflist) = len(reglist) or len(coefflist) * numcol = len(reglist)
    numcol needs to be specified to get the number of columns in the matrices.

    I also specify that paramlist = [] so that I can use the returndict directly in gettabular_matrixdict.
    """
    import copy
    import sys

    numrow = len(modelfitlist) / numcol

    betalist = [None] * len(modelfitlist)
    selist = [None] * len(modelfitlist)
    pvallist = [None] * len(modelfitlist)
    for i in range(len(coefflist)):
        if coefflist[i] in modelfitlist[i].params:
            betalist[i] = modelfitlist[i].params[coefflist[i]]
            selist[i] = modelfitlist[i].bse[coefflist[i]]
            pvallist[i] = modelfitlist[i].pvalues[coefflist[i]]


    numrow = int(len(modelfitlist) / numcol)

    betamatrix = [betalist[numcol * i : numcol * (i + 1)] for i in range(numrow)]
    sematrix = [selist[numcol * i : numcol * (i + 1)] for i in range(numrow)]
    pvalmatrix = [pvallist[numcol * i : numcol * (i + 1)] for i in range(numrow)]

    return(betamatrix, pvalmatrix, sematrix)

# Test full:{{{1
def example_all():
    if not os.path.isdir(__projectdir__ / Path('temp/regrun/')):
        os.makedirs(__projectdir__ / Path('temp/regrun/'))

    print('extenddf test')
    extenddf_test()

    print('\ngenerate ols matrices, run ols and print summary: y same, X different, df same')
    runols_test_ysame_dfsame()

    print('\ngenerate ols matrices, run ols and print summary: y diff, X diff, df diff')
    runols_test_alldiff()

    print('\nget coeff list and betas')
    getsamecoeffmatrices_test()

    print('\nget parammatrix')
    getparammatrix_test()

    print('\nmultireg_tabular')
    getmultiregtabular_test_tabular()

    print('\nmultireg fewer coefficients shown in table')
    getmultiregtabular_test_xcoeffreduce()

    print('\nmultireg show different parameters in table')
    getmultiregtabular_test_paramchange()

    print('\n multireg_tabular')
    getmultiregtabular_test_tabsecmerge()

