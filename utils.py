#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:17:49 2020

@author: fan
"""
#%%
import numpy as np
from numpy.linalg import inv
from numpy.random import choice
from scipy.special import logit, expit
from scipy.stats import multivariate_normal, norm, truncnorm
from scipy.stats import wishart#, invwishart
from scipy.stats import dirichlet
from sklearn.cluster import KMeans

#import pandas as pd
from copy import copy

#%%

# 1-d Gaussian stuff (score model)

## linked score

def initializeLinkedScore(L, initThres = 0.6):
    '''
    Initialize things based on thresholding on the linked score;
    Returns:
        - logit transformed link score
        - indices of pairs that are selected in the point processes
        - initial value of muL, gammaL (inverse of sigma^2_l)
    L: length N linked scores of all pairs
    '''
    
    inds = np.where(L > initThres)[0]
    
    L = logit(L)
    Thres = logit(initThres)
    
    muL = np.mean(L[L > Thres])
    
    deMean = np.where(L > Thres, L - muL, L)
    
    gammaL = 1/np.mean(deMean ** 2)
    
    return L, inds, muL, gammaL


def updateLModel(L, E_MF, E_FM, muL, gammaL, gammaPrior):
    '''
    Update linked score model (muL and gammaL) given the point configurations
    Returns muL and gammaL
    L: length N linked scores (transformed) of all pairs
    E_MF: dictionary of (a_M, a_F) points in the MF process, key is pair index
    E_FM: dictionary of (a_F, a_M) points in the Fm process, key is pair index
    gammaPrior: a dictionary of prior for gammaL, "nu0" and "sigma0"
    '''
    
    inds = list(E_MF.keys()) + list(E_FM.keys())
    
    mu_mean = np.mean(L[inds])
    mu_std = 1/np.math.sqrt(len(inds) * gammaL)
    
    muL = truncnorm(a=(0-mu_mean)/mu_std, b=np.inf).rvs() * mu_std + mu_mean
    
    #deMean = L
    deMean = copy(L)
    deMean[inds] = deMean[inds] - muL
    SS = np.sum(deMean ** 2)
    
    gammaL = np.random.gamma((gammaPrior['nu0'] + len(L))/2, 
                             2/(gammaPrior['nu0'] * gammaPrior['sigma0'] + SS))
    
    return muL, gammaL

def evalLLikelihood(L, E_MF, E_FM, muL, gammaL, subset=None, log=True):
    '''
    Evaluate the linked score component of the likelihood (on a subset of entries);
    Returns length len(L) (or len(subset)) array of (log-)likelihood
    L: length N linked scores (transformed) of all pairs
    E_MF: dictionary of (a_M, a_F) points in the MF process, key is pair index
    E_FM: dictionary of (a_F, a_M) points in the Fm process, key is pair index
    muL, gammaL: parameters of the L model
    subset: list of SORTED indices (if None, then evaluate likelihood on all entries)
    log: bool, output log-likelihood?
    '''
    # get the indices in either point process
    inds= list(E_MF.keys()) + list(E_FM.keys())
    if subset is not None:
        indsIn = list(set(subset) & set(inds))
        indsOut = list(set(subset) - set(inds))
        res = np.empty(len(subset))
        indices = np.array(subset)
    else:
        indices = np.array(range(len(L)))
        indsIn = inds
        indsOut = list(set(indices) - set(inds))
        res = np.empty(len(L))
        
    sd = 1/np.math.sqrt(gammaL)  
    #logDensIn = norm(loc=muL, scale=sd).logpdf(L[indsIn]) if len(indsIn) > 0 else 
    if len(indsIn) > 0:
        logDensIn = norm(loc=muL, scale=sd).logpdf(L[indsIn])
        res[np.searchsorted(indices, indsIn)] = logDensIn
    if len(indsOut) > 0:
        logDensOut = norm(loc=0, scale=sd).logpdf(L[indsOut])
        res[np.searchsorted(indices, indsOut)] = logDensOut
        
    if not log:
        res = np.exp(res)
        
    return res
        

## direction score
    
def initializeDirectScore(D, inds):
    '''
    Initialize direction score stuff based on thresholding results of linked score;
    Returns:
        - logit transformed direction score
        - indices of pairs that are selected in each point process
        - initial value of muNegD, muD, gammaD (inverse of sigma^2_d)
    D: length N linked scores of all pairs (in same order as L)
    '''
    
    D = logit(D)
    
    inds = set(inds)
    indsMF = inds & set(np.where(D > 0)[0])
    indsFM = inds - indsMF
    
    indsMF = list(indsMF)
    indsFM = list(indsFM)
    
    muD = np.mean(D[indsMF])
    muNegD = np.mean(D[indsFM])
    
    Dsel = D[list(inds)]
    deMean = np.where(Dsel > 0, Dsel-muD, Dsel-muNegD)
    gammaD = 1/np.mean(deMean ** 2)
    
    return D, indsMF, indsFM, muD, muNegD, gammaD


def updateDModel(D, E_MF, E_FM, muD, muNegD, gammaD, gammaPrior):
    '''
    Update linked score model (muL and gammaL) given the point configurations
    Returns muD, muNegD, gammaD
    D: length N MF-direction scores (transformed) of all pairs
    E_MF: dictionary of (a_M, a_F) points in the MF process, key is pair index
    E_FM: dictionary of (a_F, a_M) points in the Fm process, key is pair index
    gammaPrior: a dictionary of prior for gammaL, "nu0" and "sigma0"
    '''
    
    indsMF = list(E_MF.keys()) 
    indsFM = list(E_FM.keys())
    
    muD_mean = np.mean(D[indsMF])
    muD_std = 1/np.math.sqrt(len(indsMF) * gammaD)
    muD = truncnorm(a=(0-muD_mean)/muD_std, b=np.inf).rvs() * muD_std + muD_mean
    
    muNegD_mean = np.mean(D[indsFM])
    muNegD_std = 1/np.math.sqrt(len(indsFM) * gammaD)
    muNegD = truncnorm(a=-np.inf, b=(0-muNegD_mean)/muNegD_std).rvs() * muNegD_std + muNegD_mean
    
    #deMean = D
    deMean = copy(D)
    deMean[indsMF] = deMean[indsMF] - muD
    deMean[indsFM] = deMean[indsFM] - muNegD
    SS = np.sum(deMean ** 2)
    
    gammaD = np.random.gamma((gammaPrior['nu0'] + len(D))/2, 
                             2/(gammaPrior['nu0'] * gammaPrior['sigma0'] + SS))
    
    return muD, muNegD, gammaD


def evalDLikelihood(D, E_MF, E_FM, muD, muNegD, gammaD, subset=None, log=True):
    '''
    Evaluate the direction score component of the likelihood (on a subset of entries);
    Returns length len(D) (or len(subset)) array of (log-)likelihood
    D: length N direction scores (transformed) of all pairs
    E_MF: dictionary of (a_M, a_F) points in the MF process, key is pair index
    E_FM: dictionary of (a_F, a_M) points in the FM process, key is pair index
    muD, muNegD, gammaD: parameters of the D model
    subset: list of SORTED indices (if None, then evaluate likelihood on all entries)
    log: bool, output log-likelihood?
    '''
    # get the indices in each point process
    indsMF= list(E_MF.keys())
    indsFM = list(E_FM.keys())
    
    # get indices in MF, MF and out
    if subset is not None:
        indsMF = list(set(subset) & set(indsMF))
        indsFM = list(set(subset) & set(indsFM))
        indsOut = list(set(subset) - (set(indsMF) | set(indsFM)))
        res = np.empty(len(subset))
        indices = np.array(subset)
    else:
        indices = np.array(range(len(D)))
        indsOut = list(set(indices) - (set(indsMF) | set(indsFM)))
        res = np.empty(len(D))
        
    sd = 1/np.math.sqrt(gammaD)  
    
    #print(indices)
    
    #print((set(indsMF) | set(indsMF)), indsOut)

    if len(indsMF) > 0:
        logDensMF = norm(loc=muD, scale=sd).logpdf(D[indsMF])
        res[np.searchsorted(indices, indsMF)] = logDensMF
    if len(indsFM) > 0:
        logDensFM = norm(loc=muNegD, scale=sd).logpdf(D[indsFM])
        res[np.searchsorted(indices, indsFM)] = logDensFM
    if len(indsOut) > 0:
        logDensOut = norm(loc=0, scale=sd).logpdf(D[indsOut])
        res[np.searchsorted(indices, indsOut)] = logDensOut
        
    if not log:
        res = np.exp(res)
        
    return res

#%%
   
# test score model update
    
if __name__ == '__main__':
#    ## test initialization
#    L = (1-0.3)* np.random.random_sample(100) + 0.3
#    D = np.random.random_sample(100)
#    
#    Ltrans, inds, muL, gammaL = initializeLinkedScore(L, initThres = 0.6)
#    Dtrans, indsMF, indsFM, muD, muNegD, gammaD = initializeDirectScore(D, inds)
#        
#    ## test update
#    gaPrior = {'nu0': 2, 'sigma0': 1}
#    ## completely made up points...
#    E_MF = dict(zip(indsMF, np.random.random_sample(len(indsMF))))
#    E_FM = dict(zip(indsFM, np.random.random_sample(len(indsFM))))
#    
#    print(updateLModel(Ltrans, E_MF, E_FM, muL, gammaL, gaPrior))
#    print(updateDModel(Dtrans, E_MF, E_FM, muD, muNegD, gammaD, gaPrior))
    
    Ltrans, inds, muL, gammaL = initializeLinkedScore(L, initThres = 0.6)
    Dtrans, indsMF, indsFM, muD, muNegD, gammaD = initializeDirectScore(D, inds)
    
    print(Ltrans)
    print(Dtrans)
    
    E_MF = {i:v for i,v in E.items() if i in range(50)}
    E_FM = {i:v for i,v in E.items() if i in range(50,100)}
    
    gaPrior = {'nu0': 2, 'sigma0': 1}
    
    maxIter = 1000
    
    params = {'muL': [], 'gammaL':[], 'muD': [], 'muNegD': [], 'gammaD': []}
    
    for it in range(maxIter):
        muL, gammaL = updateLModel(Ltrans, E_MF, E_FM, muL, gammaL, gaPrior)
        params['muL'].append(muL); params['gammaL'].append(gammaL)
        
        muD, muNegD, gammaD = updateDModel(Dtrans, E_MF, E_FM, muD, muNegD, gammaD, gaPrior)
        params['muD'].append(muD); params['muNegD'].append(muNegD)
        params['gammaD'].append(gammaD)

    print(Ltrans)
    print(Dtrans)

#%%

# The point process stuff

def initializePP(E, indsMF, indsFM):
    '''
    Initialize MF and FM point process configurations.
    Returns:
        - E_MF, E_FM: dictionary of (a_source, a_recipient) points
        - gammaMF, gammaFM: initial values of the scales
    E: dictionary of all (a_M, a_F) points (for all the pairs in data)
    indsMF, indsFM: some assignment of indices in MF and FM surfaces
    '''
    
    E_MF = {pair: age for pair, age in E.items() if pair in indsMF}
    E_FM = {pair: age[::-1] for pair, age in E.items() if pair in indsFM}
    
    gammaMF = len(indsMF)
    gammaFM = len(indsFM)
    
    return E_MF, E_FM, gammaMF, gammaFM

def updateGammaPP(E_MF, E_FM, gammaPrior):
    '''
    Update gammaMF and gammaFM
    gammaPrior: dictionary of prior, with "n0" and "b0"
    '''
    
    N_MF = len(E_MF); N_FM = len(E_FM)
    
    gammaMF = np.random.gamma(gammaPrior['n0']+N_MF, 
                              1/(gammaPrior['b0']+1))
    gammaFM = np.random.gamma(gammaPrior['n0']+N_FM, 
                              1/(gammaPrior['b0']+1))

    return gammaMF, gammaFM
        

def proposePPoldold(E, E_MF, E_FM, batch_size = 1):
    '''
    (The oldest version of the function)
    Propose change in the point process configurations.
    With probability 1/3, select [batch_size] pairs to 
        1) birth, 2) death, or 3) swap
    Returns:
        - updated E_MF, E_FM
        - SORTED indices of the pairs that are changed
        - the type of change they went through
    '''
    N = len(E)
    indsMF = list(E_MF.keys())
    indsFM = list(E_FM.keys())
    indsOut = list(set(range(N)) - set(indsMF+indsFM))
    
    step = choice(['b','d','s'])
    
    if step == 'b':
        # birth step
        if len(indsOut) <= batch_size:
            chosen = indsOut
        else:
            chosen = choice(indsOut, size=batch_size, replace=False)
        # for each chosen one, randomly assign it to each surface
        for c in chosen:
            if np.random.random_sample() < 0.5:
                E_MF[c] = E[c]
            else:
                E_FM[c] = E[c][::-1] # need to reverse age order on FM
    elif step == 'd':
        # death step
        if N - len(indsOut) <= batch_size:
            chosen = indsMF + indsFM
        else:
            chosen = choice(indsMF+indsFM, size=batch_size, replace=False)
        # for each chosen one, delete it from its event set
        for c in chosen:
            if c in E_MF:
                del E_MF[c]
            else:
                del E_FM[c]
    else:
        # swap step
        if N - len(indsOut) <= batch_size:
            chosen = indsMF + indsFM
        else:
            chosen = choice(indsMF+indsFM, size=batch_size, replace=False)
        # for each chosen one, switch its surface (and reverse age order)
        for c in chosen:
            if c in E_MF:
                age_c = E_MF.pop(c)
                E_FM[c] = age_c[::-1]
            else:
                age_c = E_FM.pop(c)
                E_MF[c] = age_c[::-1]
                
    chosen = list(chosen)
    chosen.sort()
    
    #print('Change type in this proposal: {}.'.format(step))
                
    return E_MF, E_FM, chosen, step


def proposePPold(E, E_MF, E_FM, batch_size = 1):
    '''
    (Older version of the "proposePP" function)
    Propose change in the point process configurations.
    Uniformly select [batch_size] pairs to 
        1) birth, 2) death, or 3) swap
        each with 1/3 probability
    Returns:
        - updated E_MF, E_FM
        - SORTED indices of the pairs that are changed
        - the type of change they went through
    '''
    N = len(E)
    #indsMF = list(E_MF.keys())
    #indsFM = list(E_FM.keys())
    #indsOut = list(set(range(N)) - set(indsMF+indsFM))
    indsOut = set(range(N)) - (set(E_MF.keys()) | set(E_FM.keys()))
    
    # randomly select the indices of our batch
    batch = choice(range(N), batch_size, replace=False)
    # assign each of them with a type of change
    steps = choice(['b','d','s'], batch_size, replace=True)
    
    for ind, step in zip(batch,steps):
        if step == 'b' and ind in indsOut:
            # birth step: only works for those outside of the event sets
            # with probability 1/2, add it in MF or FM
            if np.random.random_sample() < 0.5:
                E_MF[ind] = E[ind]
            else:
                E_FM[ind] = E[ind][::-1] # need to reverse age order on FM
        elif step == 'd' and ind not in indsOut:
            # death step: only works for those inside the event sets
            if ind in E_MF:
                del E_MF[ind]
            else:
                del E_FM[ind]
        elif step == 's' and ind not in indsOut:
            # swap step: only works for those inside the event sets
            if ind in E_MF:
                age_ind = E_MF.pop(ind)
                E_FM[ind] = age_ind[::-1]
            else:
                age_ind = E_FM.pop(ind)
                E_MF[ind] = age_ind[::-1]
        else:
            continue
                
    chosen = list(batch)
    chosen.sort()
    
    steps = "-".join(list(steps))
                
    return E_MF, E_FM, chosen, steps
    


def proposePP(E, E_MF, E_FM, batch_size = 1):
    '''
    (The lastest version of the function)
    Propose change in the point process configurations.
    With equal probability, select only ONE event from the potential change events:
        1) birth to either E_MF or E_FM,
        2) death from either E_MF or E_FM,
        2) swap between E_MF and E_FM 
    Returns:
        - updated E_MF, E_FM
        - THE index of the pair that is changed
        - the type of change they went through
        
    NOTE: since some of the potential changes conflict, right now only ONE change is proposed
    That is, batch_size is FIXED at 1 for now!
    '''
    N = len(E)
    indsMF = list(E_MF.keys()); N_MF = len(indsMF)
    indsFM = list(E_FM.keys()); N_FM = len(indsFM)
    indsOut = list(set(range(N)) - set(indsMF+indsFM)); N_out = N - N_MF - N_FM
    
    change_prob = np.array([N_out * 2, N_MF + N_FM, N_MF + N_FM])
    change_prob = change_prob/change_prob.sum()
    
    step = choice(['b','d','s'], p = change_prob)
    
    if step == 'b':
        # birth step
        if N_out <= batch_size:
            chosen = indsOut
        else:
            chosen = choice(indsOut, size=batch_size, replace=False)
        # for each chosen one, randomly assign it to each surface
        for c in chosen:
            if np.random.random_sample() < 0.5:
                E_MF[c] = E[c]
            else:
                E_FM[c] = E[c][::-1] # need to reverse age order on FM
    elif step == 'd':
        # death step
        if N_MF + N_FM <= batch_size:
            chosen = indsMF + indsFM
        else:
            chosen = choice(indsMF+indsFM, size=batch_size, replace=False)
        # for each chosen one, delete it from its event set
        for c in chosen:
            if c in E_MF:
                del E_MF[c]
            else:
                del E_FM[c]
    else:
        # swap step
        if N_MF + N_FM <= batch_size:
            chosen = indsMF + indsFM
        else:
            chosen = choice(indsMF+indsFM, size=batch_size, replace=False)
        # for each chosen one, switch its surface (and reverse age order)
        for c in chosen:
            if c in E_MF:
                age_c = E_MF.pop(c)
                E_FM[c] = age_c[::-1]
            else:
                age_c = E_FM.pop(c)
                E_MF[c] = age_c[::-1]
                
    chosen = list(chosen)
    chosen.sort()
    
    #print('Change type in this proposal: {}.'.format(step))
                
    return E_MF, E_FM, chosen, step


def getPoints(E, subset=None):
    '''
    Return a (n,p) array of the points in event set E (or a subset)
    E: dictionary of indice, age pair
    subset: list of subset indices
    '''
    if not E:
        # if E is empty, raise an Error
        raise ValueError('The point event set is empty!')
    else:
        #p = X.shape[1]
        if subset:
            E_sub = {i: age for i,age in E.items() if i in subset}
            X = np.array(list(E_sub.values()))
            #n = len(subset)
        else:
            X = np.array(list(E.values()))
            #n = len(E)
        #X = X.reshape((n,p))

    return X

#%%
if __name__ == '__main__':
    E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
    X = getPoints(E)
    print(X.shape)
    
    inds1 = choice(range(100), size=38, replace=False)
    inds2 = choice(list(set(range(100)) - set(inds1)), size = 30, replace=False)
    
    E1, E2, gam1, gam2 = initializePP(E, inds1, inds2)
    
    E1, E2, chosen = proposePP(E, E1, E2, 10)

#%%

# Gaussian mixture stuff (spatal density model)

def initializeGMM(X, K=2):
    '''
    Initialize a finite Gaussian mixture model via k-means;
    Returns components (mean and precision matrix) and component labels
    X: (n,p) array of data
    K: number of components
    '''
    kmeans = KMeans(n_clusters=K).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    components = list()
    for k in range(K):
        components.append((centers[k,:], np.cov(X[labels==k,:],rowvar=False)))
        
    return components, labels


def updateOneComponent(X, mu, precision, muPrior, precisionPrior):
    '''
    X: (n,p) array of data
    mu: (p,1) array of current mean
    precision: (p,p) matrix of current precision
    muPrior: dictionary of prior mean and precision
    precisionPrior: dictionary of prior df and invScale
    '''
    
    n = X.shape[0]
    An_inv = inv(muPrior['precision'] + n * precision)
    Xsum = np.sum(X, axis=0)
    bn = muPrior['precision'].dot(muPrior['mean']) + precision.dot(Xsum)
    
    mu = multivariate_normal(An_inv.dot(bn), An_inv).rvs()
    
    S_mu = np.matmul((X-mu).T, X-mu)
    
    precision = wishart(precisionPrior['df'] + n, 
                        inv(precisionPrior['invScale'] + S_mu)).rvs()
    
    return mu, precision

def updateGaussianComponents(X, Z, components, muPrior, precisionPrior):
    '''
    X: (n,p) array of data
    Z: length n, array like component indicator
    components: list of (mu, precision) for K Gaussian components
    muPrior: dictionary of prior mean and precision
    precisionPrior: dictionary of prior df and invScale
    '''
    K = len(components)
    
    for k in range(K):
        subX = X[Z==k,:]
        if subX.shape[0] > 0:
            mu, precision = components[k]
            components[k] = updateOneComponent(subX, mu, precision, 
                      muPrior, precisionPrior)
            
    return components

def getProbVector(p):
    p = np.exp(p - np.max(p))
    #print(p)
    return p/p.sum()

def updateComponentIndicator(X, weight, components):
    '''
    X: (n,p) array of data
    components: list of (mu, precision) for K Gaussian components
    (05/13 fix: use weights in indicator update! previous version was wrong)
    '''
    K = len(components)
    n = X.shape[0]
    
    logDens = np.empty((K,n))
    
    for k in range(K):
        mu, precision = components[k]
        MVN = multivariate_normal(mu, inv(precision))
        logDens[k,:] = MVN.logpdf(X) + np.log(weight[k])
#        logProb = MVN.logpdf(X)
#        if np.any(np.isnan(logProb)):
#            print(mu, precision)
#            raise ValueError("NaN in log likelihood!")
#        else:
#            logDens[k,:] = logProb
        
    Z = np.apply_along_axis(lambda v: choice(range(K), replace=False, 
                                             p=getProbVector(v)), 0, logDens)
    return Z

def updateMixtureWeight(Z, weightPrior):
    '''
    Z: length n, array like component indicator
    weightPrior: length K, array like prior (for the Dirichlet prior)
    '''
    unique, counts = np.unique(Z, return_counts=True)
    mixtureCounts = dict(zip(unique,counts))
    
    alpha = weightPrior
    
    for k in mixtureCounts:
        alpha[k] += mixtureCounts[k]
        
    return dirichlet(alpha).rvs()[0]

def evalDensity(X, weight, components, log=True):
    '''
    Evaluate the entire density function (after mixture) on points X;
    Returns a length-n array of density/log-density
    X: (n,p) array of data
    weight: length K vector of mixture weights
    components: list of (mu, precision) for K Gaussian components
    '''
    
    n = X.shape[0]
    K = len(weight)
    
    mix_dens = np.empty((n,K))
    
    for k in range(K):
        mu, precision = components[k]
        MVN = multivariate_normal(mu, inv(precision))
        mix_dens[:,k] = MVN.pdf(X)
        
    #print(mix_dens)
        
    total_dens = np.sum(weight * mix_dens, axis=1)
    
    if log:
        total_dens = np.log(total_dens)
        
    return total_dens
#%% test
#x_test = np.random.randn(50,2) + 2

# initialize function
#components, Z = initializeGMM(x_test)

# update component function
#muP = {'mean': np.array([0,0]), 'precision': np.eye(2)}
#preP = {'df': 2, 'invScale': np.eye(2)*.01}
#
#updateOneComponent(x_test, np.array([0.1,0.1]), np.eye(2), muP, preP)

# update indicator function
#components = [(np.zeros(2), np.eye(2)), (np.ones(2), np.eye(2) * 0.01)]
#Z = updateComponentIndicator(x_test, components)
#Z.shape[0] == x_test.shape[0]

# update mixture weight function
#updateMixtureWeight(Z, np.ones(2))


#%% test out the whole process
if __name__ == "__main__":

    from time import perf_counter
    
    muP = {'mean': np.array([0,0]), 'precision': np.eye(2)}
    preP = {'df': 2, 'invScale': np.eye(2)*.0001}
    weightP = np.ones(2)
    
    x_1 = np.random.randn(100,2) + 10
    x_2 = np.random.randn(100,2) -10
    x_test = np.concatenate((x_1,x_2),axis=0)
    
    tic = perf_counter()
    
    components, Z = initializeGMM(x_test)
    
    maxIter = 100
    
    for i in range(maxIter):
        components = updateGaussianComponents(x_test, Z, components, 
                                              muP, preP)
        Z = updateComponentIndicator(x_test, components)
        w = updateMixtureWeight(Z, weightP)
        
    #log_dens = evalDensity(x_test[:10,:], w, components)
    #print("log likelihood of first 10 points: {:.4f}".format(log_dens))
    
    #print(evalDensity(x_test, w, components))
        
    elapsed = perf_counter() - tic
    
    print("Total time {:.4f} seconds, with {:.4f} seconds per iteration.".format(elapsed,elapsed/maxIter))
        
    # It seems to work...
    # But occassionally would encounter NaN in the log density??
    # Probably fixed...
    
#%%
# re-rest the Gaussian mixture model
#from time import perf_counter
#    
#muP = {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001}
#preP = {'df': 2, 'invScale': np.eye(2)}
#weightP = np.ones(3)
#
#x_test = X[:100,:]
#
#tic = perf_counter()
#
#components, Z = initializeGMM(x_test,K=3)
#
#maxIter = 2000
#
#for i in range(maxIter):
#    components = updateGaussianComponents(x_test, Z, components, 
#                                          muP, preP)
#    Z = updateComponentIndicator(x_test, components)
#    w = updateMixtureWeight(Z, weightP)
#    
##log_dens = evalDensity(x_test[:10,:], w, components)
##print("log likelihood of first 10 points: {:.4f}".format(log_dens))
#
##print(evalDensity(x_test, w, components))
#    
#elapsed = perf_counter() - tic
#
#print("Total time {:.4f} seconds, with {:.4f} seconds per iteration.".format(elapsed,elapsed/maxIter))

#%%
# functions to simulate data
def simulateGMM(N, weight, components):
    comp_counts = np.random.multinomial(N, weight)
    data = None
    for k in range(len(weight)):
        if comp_counts[k] > 0:
            data_k = np.random.multivariate_normal(components[k][0], inv(components[k][1]), comp_counts[k])
            if data is None:
                data = data_k
            else:
                data = np.vstack((data,data_k))
    return data

def simulateLatentPoissonGMM(N, Settings):
    '''
    Simulate a dataset with N pairs
    Return: E, L, D
    Settings: a giant dictionary with settings and parameters
        - 'N_MF', 'N_FM': number of points in each point process
        - 'muD', 'muNegD', 'muL': the score model means
        - 'gammaD', 'gammaL': the score model precisions (inverse variance)
        - 'componentsMF', 'componentsFM': length K list of GMM components (mean vector, precision matrix)
        - 'weightMF', 'weightFM': mixture weight of GMM on each process
    '''
    
    N_MF = Settings['N_MF']
    N_FM = Settings['N_FM']
    N_out = N - N_FM - N_MF
    
    assert N_MF + N_FM <= N
    
    # 1. Generate L and D
    Lin = norm(loc=Settings['muL'], scale = 1/np.sqrt(Settings['gammaL'])).rvs(N_MF + N_FM)
    Lout = norm(loc=0, scale = 1/np.sqrt(Settings['gammaL'])).rvs(N_out)
    L = expit(np.concatenate((Lin, Lout)))
    
    D_MF = norm(loc=Settings['muD'], scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_MF)
    D_FM = norm(loc=Settings['muNegD'], scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_FM)
    D_out = norm(loc=0, scale = 1/np.sqrt(Settings['gammaD'])).rvs(N_out)
    D = expit(np.concatenate((D_MF,D_FM,D_out)))
    
    # 2. Generate E
    ## Those who are in MF
    MFvalues = simulateGMM(N_MF, Settings['weightMF'], Settings['componentsMF'])
    Evalues = list(MFvalues)
    
    ## Those who are in FM
    FMvalues = simulateGMM(N_FM, Settings['weightFM'], Settings['componentsFM'])
    FMvalues = FMvalues[:,::-1] # flip the age, so that it's always (a_M, a_F)
    Evalues.extend(list(FMvalues))

    ## Those who are outside
    ### randomly sample within the range of points already sampled
    Mins = np.min(Evalues, axis=0)
    Maxs = np.max(Evalues, axis=0)
    AgeM = np.random.random_sample((N_out,1)) * (Maxs[0] - Mins[0]) + Mins[0]
    AgeF = np.random.random_sample((N_out,1)) * (Maxs[1] - Mins[1]) + Mins[1]
    Evalues.extend(list(np.hstack((AgeM,AgeF))))
    
    ## put together
    E = dict(zip(range(N),Evalues))
    
    return E, L, D
    