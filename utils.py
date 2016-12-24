import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def poisson_pseudoR2(y, y_hat, y_null=None, verbose=False,safe=True):
    """
    Determines how well a model does at predicting outcomes compared to the mean of the training set.

    Parameters:
    ==========
    y = predicted variable
    y_hat = model
    y_null = null hypothesis, mean of predicted variable in training set.

    Returns:
    ==========
    R2 : pseudo r2 values
    """

    if y_null is None:
        y_null = np.mean(y)
    if np.ndim(y_hat) > 1:
        #extremely important line of code
        y_hat = np.squeeze(y_hat)
        if verbose == True:
            print "y_hat squeezed"
    if np.ndim(y) > 1:
        #extremely important line of code
        y = np.squeeze(y)
        if verbose == True:
            print "y_test squeezed"

    eps = np.spacing(1)
    L1 = np.sum(y*np.log(eps+y_hat) - y_hat)
    if any(np.isnan(y*np.log(eps+y_hat) - y_hat)):
        if safe == True:
            print 'nan found. Returning nan'
            return np.nan
        else:
            print "nan's found in L1. Using nanmean"
            L1 = np.nansum(y*np.log(eps+y_hat) - y_hat)
    L0 = np.sum(y*np.log(eps+y_null) - y_null)
    if any(np.isnan(y*np.log(eps+y_hat) - y_hat)):
        if safe == True:
            print 'nan found. Returning nan'
            return np.nan
        else:
            print "nan's found in L0. Using nanmean"
            L0 = np.nansum(y*np.log(eps+y_null) - y_null)

    LS = np.sum(y*np.log(eps+y) - y)
    R2 = 1-(LS-L1)/(LS-L0)

    if verbose:
        print "y " + str(y.shape)
        print "y_hat" + str(y_hat.shape)
        print "L1 "+str(L1)
        print "L0 "+str(L0)
        print "LS "+str(LS)
        print "R2 " + str(R2)

    return R2