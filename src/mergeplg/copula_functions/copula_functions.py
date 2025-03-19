#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:00:23 2024

@author: erlend

Many functions are just copied from RMWSPy
"""

import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#import scipy.spatial as sp
import scipy.interpolate as interpolate
import scipy.stats as st

import datetime

from mergeplg.copula_functions.gcopula_sparaest import paraest_multiple_tries

def calculate_marginal(prec, cml_prec=None):
    """
    Calculate marginal distribution either by rain observed by gauges (prec) only,
    or including the high CML values also if they exceed the gauge values.
    """

    if cml_prec is not None:
        # add the cml_prec that are higher than prec
        hcmlp = np.copy(cml_prec) * 10.0
        hcmlp = hcmlp[hcmlp > prec.max()]
        prec = np.concatenate((prec, hcmlp))

    # fit a non-parametric marginal distribution using KDE with Gaussian kernel
    # this assumes that there are wet observations
    p0 = 1.0 - float(prec[prec > 0].shape[0]) / prec.shape[0]

    if len(prec[prec > 0]) < 5:
        cv = 2
    else:
        cv = 5

    # optimize the kernelwidth
    prec_wet = np.log(prec[prec > 0])
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=cv
    )
    grid.fit(prec_wet[:, None])

    # use optimized kernel for kde
    kde = KernelDensity(bandwidth=grid.best_params_["bandwidth"], kernel="gaussian")
    kde.fit(prec_wet[:, None])

    # build cdf and invcdf from pdf
    xx = np.arange(prec_wet.min() - 1.0, prec_wet.max() + 1.0, 0.001)
    logprob = np.exp(kde.score_samples(xx[:, None]))
    cdf_ = np.cumsum(logprob) * 0.001
    cdf_ = np.concatenate(([0.0], cdf_))
    cdf_ = np.concatenate((cdf_, [1.0]))

    xx = np.concatenate((xx, [prec_wet.max() + 1.0]))
    xx = np.concatenate(([prec_wet.min() - 1.0], xx))
    cdf = interpolate.interp1d(xx, cdf_, bounds_error=False)
    invcdf = interpolate.interp1d(cdf_, xx)

    # marginal distribution variables
    marginal = {}
    marginal["p0"] = p0
    marginal["cdf"] = cdf
    marginal["invcdf"] = invcdf

    return marginal

def calculate_copula(
    yx, prec, outputfile=None, covmods='exp', ntries=6, nugget=0.05,  mode = None, maxrange = 100, minrange = 1, p0 = None
):
    """
    Wrapper function for copula / spatial dependence calculation
    """


    # transform to rank values
    # erlend: this makes the distribution flat, like in the KDE, thus it can 
    # be transformed to a normal space using the copula ( normal copula is
    # applied later )
    u = (st.rankdata(prec) - 0.5) / prec.shape[0]
    
    if p0 is not None:
        ind = u > p0*0.5
        u = u[ind]
        yx = yx[ind]
    
    
    # set subset size
    if len(prec[prec > 0]) < 5:
        n_in_subset = 2
    else:
        n_in_subset = 5

    # calculate copula models
    cmods = paraest_multiple_tries(
        np.copy(yx),
        u,
        ntries=[ntries, ntries],
        n_in_subset=n_in_subset,
        # number of values in subsets
        neighbourhood="random",
        # subset search algorithm
        maxrange = maxrange,
        minrange = minrange,        
        covmods=[covmods],  # covariance functions
        outputfile=outputfile,
    )  # store all fitted models in an output file

    # take the copula model with the highest likelihood
    # reconstruct from parameter array
    likelihood = -666
    for model in range(len(cmods)):
        for tries in range(ntries):
            if cmods[model][tries][1] * -1.0 > likelihood:
                likelihood = cmods[model][tries][1] * -1.0
                #                 cmod = "0.05 Nug(0.0) + 0.95 %s(%1.3f)" % (
                #                     covmods[model], cmods[model][tries][0][0])
                cmod = "%1.3f Nug(0.0) + %1.3f %s(%1.3f)" % (
                    nugget,
                    1 - nugget,
                    covmods[model],
                    cmods[model][tries][0][0],
                )
                if covmods[model] == "Mat":
                    cmod += "^%1.3f" % (cmods[model][tries][0][1])


    return cmod



def get_linear_constraints(yx, prec, marginal):
    """
    Transform observations to standard normal using the fitted cdf;
    """

    # zero (dry) observations
    mp0 = prec == 0.0
    lecp = yx[mp0]
    lecv = np.ones(lecp.shape[0]) * st.norm.ppf(marginal["p0"])

    # wet observations
    yx_wet = yx[~mp0]
    prec_wet = np.log(prec[~mp0])

    # non-zero (wet) observations
    cp = yx_wet
    cv = st.norm.ppf(
        (1.0 - marginal["p0"]) * marginal["cdf"](prec_wet) + marginal["p0"]
    )

    # delete NaNs that appear when values are out of the interpolation range of cdf
    ind_nan = np.where(np.isnan(cv))
    cp = np.delete(cp, ind_nan, axis=0)
    cv = np.delete(cv, ind_nan, axis=0)

    #     lin_data = np.delete(lin_data, ind_nan, axis=0)

    return cp, cv, lecp, lecv


def backtransform(data, marginal):
    """
    Transform from standard normal to acutal value space.
    """

    rain = st.norm.cdf(data)
    mp0 = rain <= marginal["p0"]
    rain[mp0] = 0.0
    rain[~mp0] = (rain[~mp0] - marginal["p0"]) / (1.0 - marginal["p0"])
    rain[~mp0] = marginal["invcdf"](rain[~mp0])
    rain[~mp0] = np.exp(rain[~mp0]) / 10.0

    return rain


def get_copula_params(ds):
    """
    Copula parameters
    """

    # range and model
    cov_rng = []
    cov_mod = []

    for i in range(len(ds.time)):
        name_cop = (ds.copula.values)[i]

        if name_cop == "nan":
            cov_rng.append(np.nan)
            cov_mod.append("nan")

        else:
            name_cop = name_cop.split("+")[1].strip().split(" ")[1]
            cov_rng.append(float(name_cop[4:-1]))
            cov_mod.append(name_cop[:3])

    return cov_rng, cov_mod