# IV:{{{1
def dropmissingrows(listoflistofmatrices):
    """
    If I have Ymatrices, Xmatrices, this can allow me to drop missing values from the matrix.

    Use when missing = 'drop' not in statsmodels function.
    """
    import copy
    import numpy as np
    import pandas as pd

    numlistofmatrices = len(listoflistofmatrices)
    nummatrices = len(listoflistofmatrices[0])

    # want to put together Xmatrices[0], Ymatrices[0] etc. 
    for j in range(nummatrices):
        listofmatrices = [pd.DataFrame(listoflistofmatrices[i][j]) for i in range(numlistofmatrices)]
        # get combined dataframe
        df = pd.concat(listofmatrices, axis = 1)
        # get bad rows
        rowstodrop = df.isnull().any(axis = 1)

        # make replacement
        for i in range(numlistofmatrices):
            newmatrix = listoflistofmatrices[i][j][rowstodrop == False]

            # don't need separate argument - works same with pd and np
            # if isinstance(listoflistofmatrices[i][j], pd.DataFrame):
            #     newmatrix = listoflistofmatrices[i][j][rowstodrop == False]
            # else:
            #     newmatrix = listoflistofmatrices[i][j][rowstodrop == False]

            listoflistofmatrices[i][j] = newmatrix

    return(listoflistofmatrices)


def genivmatrices(Xmatrices, Ymatrices, Zmatrices, df = None, addconstant = True, Xappend = False, Zappend = False, extenddf = True, copydf = True):
    """
    If df is None then Yvariables and Xvariables are matrices. Otherwise, they are names.
    Yvariables can be list or just one. If just one, use this for all regressions.

    If Xappend == True then add the matrices for X together so if I have Xvariables = [['x1'], ['x2']] then actually use Xvariables = [['x1'], ['x1', 'x2']].
    If Zappend == True, do the same with Z.
    Note that for Xappend to work then I need the variables in each regression to be the same length.

    extenddf = True means that I allow for L., F. etc. when using df and copydf makes a copy of the df so the original df is unchanged.

    Equivalent to genolsmatrices
    """
    import copy
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # make copy of matrices to prevent issues with mutations of original names:
    Xmatrices = copy.deepcopy(Xmatrices)
    Ymatrices = copy.deepcopy(Ymatrices)
    Zmatrices = copy.deepcopy(Zmatrices)

    # convert Ymatrices to list if only one Y
    if not isinstance(Ymatrices, list):
        Ymatrices = [Ymatrices] * len(Xmatrices)
    # if already a list
    if len(Ymatrices) == 1 and len(Xmatrices) != 1:
        Ymatrices = [Ymatrices] * len(Xmatrices)

    # now get matrices from df if not already done:
    if df is not None:
        if extenddf is True:
            if copydf is True:
                df = copy.deepcopy(df)
            variables = list(set(Ymatrices + [i for j in Xmatrices for i in j] + [i for j in Zmatrices for i in j]))

            from regrun_func import extenddf
            df = extenddf(df, variables)

        # save Xnames since will need at the end
        ynames = Ymatrices[:]
        Xnames = Xmatrices[:]
        Znames = Zmatrices[:]

        for i in range(len(Ymatrices)):
            Ymatrices[i] = df[Ymatrices[i]]
            Xmatrices[i] = df[Xmatrices[i]]
            Zmatrices[i] = df[Zmatrices[i]]

    # now Xappend if True
    if Xappend is True:
        # want to concatenate as pandas otherwise lose variable names (just comes out as x1,x2,x3,...)
        if isinstance(Xmatrices[0], pd.DataFrame):
            for i in range(1, len(Xmatrices)):
                Xmatrices[i] = pd.concat((Xmatrices[i - 1], Xmatrices[i]), axis = 1)
                Xnames[i] = Xnames[i - 1] + Xnames[i]
        else:
            for i in range(1, len(Xmatrices)):
                Xmatrices[i] = np.concatenate((Xmatrices[i - 1], Xmatrices[i]), axis = 1)
    if Zappend is True:
        # want to concatenate as pandas otherwise lose variable names (just comes out as x1,x2,x3,...)
        if isinstance(Zmatrices[0], pd.DataFrame):
            for i in range(1, len(Zmatrices)):
                Zmatrices[i] = pd.concat((Zmatrices[i - 1], Zmatrices[i]), axis = 1)
                Znames[i] = Znames[i - 1] + Znames[i]
        else:
            for i in range(1, len(Zmatrices)):
                Zmatrices[i] = np.concatenate((Zmatrices[i - 1], Zmatrices[i]), axis = 1)
        
    # now addconstant if True:
    if addconstant is True:
        for i in range(0, len(Xmatrices)):
            Xmatrices[i] = sm.add_constant(Xmatrices[i])
            Xnames[i] = ['const'] + Xnames[i]
        for i in range(0, len(Zmatrices)):
            Zmatrices[i] = sm.add_constant(Zmatrices[i])
            Znames[i] = ['const'] + Znames[i]

    returndict = {}
    returndict['Xmatrices'] = Xmatrices
    returndict['Ymatrices'] = Ymatrices
    returndict['Zmatrices'] = Zmatrices
    returndict['Xnames'] = Xnames
    returndict['ynames'] = ynames
    returndict['Znames'] = Znames

    return(returndict)


def runiv(Xmatrices, Ymatrices, Zmatrices, dropmissing = True):
    """
    Get result.fit() for an IV regression for each element of the matrices.
    Equivalent to runols.
    """
    from statsmodels.sandbox.regression.gmm import IV2SLS

    results = []

    if dropmissing is True:
        from regrun_func import dropmissingrows
        Xmatrices, Ymatrices, Zmatrices = dropmissingrows([Xmatrices, Ymatrices, Zmatrices])
    else:
        missing = 'none'

    for i in range(len(Xmatrices)):
        result = IV2SLS(Ymatrices[i], Xmatrices[i], instrument = Zmatrices[i])
        resultfit = result.fit()
        results.append(resultfit)

    return(results)
    


def alliv(Xmatrices, Ymatrices, Zmatrices, savefolders = None, appendfolders = None, varmatrixoptions = {}, regrunoptions = {}, saveresultsoptions = {}, resultmatrixoptions = {}, tabularoptions = {}, tableoptions = {}, regrunfunction = None):
    """
    Function to run IV (only the second stage) regressions and generate summary tables and save results in one command.
    Meant to save time in writing everything.

    Call standard parse results function.
    """
    # if df is specified then we know that Ymatrices is actually a list of names.
    # if ynames not specified, use Ymatrices as the list of names.
    # need to do before generate Ymatrices since replaces name
    if varmatrixoptions['df'] is not None and 'ynames' not in varmatrixoptions:
        tabularoptions['ynames'] = Ymatrices

    # generate matrices {{{
    from regrun_func import genivmatrices
    returndict = genivmatrices(Xmatrices, Ymatrices, Zmatrices, **varmatrixoptions)
    # }}}

    # run IV{{{
    if regrunfunction is None:
        from regrun_func import runiv
        regrunfunction = runiv
    results = regrunfunction(returndict['Xmatrices'], returndict['Ymatrices'], returndict['Zmatrices'], **regrunoptions)
    # }}}

    # now call allresults
    from regrun_func import allresults
    allresults(results, savefolders = savefolders, appendfolders = appendfolders, saveresultsoptions = saveresultsoptions, resultmatrixoptions = resultmatrixoptions, tabularoptions = tabularoptions, tableoptions = tableoptions)
        
    

