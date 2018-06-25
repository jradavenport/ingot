import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import binned_statistic
from os.path import dirname,realpath


def FeH(w1w2, GJ, MG, use_MG = False):
    '''
    Predict the [Fe/H] value for stars given their (W1-W2, G-J) colors.

    Using: KNN regression, trained on a sample of stars from APOGEE (included).

    Errors estimates come from the scatter in the prediction of the
    original training data.

    Optionally (recommended!): Include the Gaia Absolute G mag in the KNN model

    Parameters
    ----------
    w1w2: float or numpy array
        the (W1-W2) WISE color of stars to estimate [Fe/H] for
    GJ: float or numpy array
        the (G-J) Gaia-2MASS color of stars to estimate [Fe/H] for
    MG: float or numpy array, optional
        the absolute G-band Gaia magnitude of stars to estimate [Fe/H] for.
        Not used by default.
    use_MG: bool, optional
        Should we use `MG` in computing the [Fe/H]? The results are slightly
        better if True, but not strictly needed. Default is False

    Returns
    -------
    FeH_preict: float or numpy array
        the predicted [Fe/H] value for these stars based on the KNN regessor
    FeH_predict_err: float or numpy array
        the estimated [Fe/H] errors for these stars based on the scatter in the
        KNN prediction of the original training data (Warning: not robust!)
    '''

    # sniff out where this code is, so can point to relative location of the required file
    dir = dirname(realpath(__file__))
    # read in the required training data that comes shipped w/ ingot
    df = pd.read_csv(dir + '/ingot_data.csv.gz')

    if use_MG:
        Xdata = np.array([df['w1'].values - df[u'w2'].values,
                         df[u'phot_g_mean_mag'].values - df[u'Jmag'].values,
                         df[u'M_G'].values]).T
        Xdata_input = np.array([w1w2, GJ, MG]).T
    else:
        Index([u'FE_H', u'Jmag', u'M_G', u'phot_g_mean_mag', u'w1', u'w2'], dtype='object')
        Xdata = np.array([df['w1'].values - df[u'w2'].values,
                         df[u'phot_g_mean_mag'].values - df[u'Jmag'].values]).T
        Xdata_input = np.array([w1w2, GJ]).T

    Ydata = df[u'FE_H'].values

    # Build the KNN model, as in the Gaia Sprint (2018) notebook
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(Xdata, Ydata)
    newY = model.predict(Xdata)

    FeH_predict = model.predict(Xdata_input)

    # use scatter to estimate errors on the prediction of the KNN
    # NOTE: this does not include observational errors, only an estimate
    stds, bes, _ = binned_statistic(Ydata,newY, statistic='std', bins=10, range=(-0.9,0.35))
    FeH_predict_err = np.interp(FeH_predict, (bes[1:]+bes[:-1])/2., stds)


    return FeH_predict, FeH_predict_err
