#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:32:17 2020

@author: fan
"""

# Aug 29, 2020
# The real 2.0 model with DP GMM


#%%
import os
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/HIV_transmission_flow')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

# set ggplot-like style
plt.style.use('ggplot')

# to set it back to default
#plt.style.use('default')

# to see all styles
# print(plt.style.available)

import pickle as pkl

#%%

# updated model from version 2.0 (2 surface, 4 types)     

# now with DP + GMM
        
from utils2_DPGMM import *

class LatentPoissonDPGMM2:
    def __init__(self, Priors, K=3, Kmax=10, linkThreshold=0.6):
        '''
        Inialize an instance of the LatentPoisson DP GMM model;
        Priors: a big dictionary of all the priors
            - "gammaPP": prior dictionary for the NHPP scale (gamma); 
                need "n0" and "b0"
            - "eta": prior dictionary for the thinning probability on each surface;
                need "a" and "b"
            - "muGMM": prior dictionary for the means in Gaussian Mixture; 
                need "mean" and "precision"
            - "precisionGMM": prior dictionary for the precision matrices in Gaussian Mixture;
                need "df" and "invScale"
            - "gammaScore": prior dictionary for inverse variance of the score models;
                need "nu0" and "sigma0"
            - "alpha": prior for the DP precision; need "a" and "b"
                
        K: num of mixture components to initialize with
        Kmax: maximum num of mixture components for the truncated DP
        
        linkThreshold: empirical lower bound to select "meaningful" linked scores
        '''
        self.name = "Latent Poisson Process with Dirichlet Process Gaussian Mixture density"
        self.linkInitialThreshold = linkThreshold
        # number of mixture components
        self.K = K
        self.Kmax = Kmax
        # prior part
        self.ScoreGammaPrior = Priors["gammaScore"]
        self.muPrior = Priors["muGMM"]
        self.precisionPrior = Priors["precisionGMM"]
        #self.weightPrior = Priors["weight"]
        self.PPGammaPrior = Priors["gammaPP"]
        self.etaPrior = Priors["eta"]
        self.alphaPrior = Priors['alpha']
        # data part
        self.E = None # all the (a_M,a_F) pairs
        self.L = None # all the linked scores
        self.D = None # all the direction scores
        self.indsMF = None # indices on the MF surface
        self.indsFM = None # indices on the FM surface
        self.inds0MF = None # indices for the ghost points on MF surface
        self.inds0FM = None # indices for the ghost points on FM surface
        #self.E_MF = None # event set on MF surface
        #self.E_FM = None # event set on FM surface
        #self.E_0 = None # event set on the outside
        # parameters
        self.muL = None
        self.muD = None
        self.muNegD = None
        self.gammaL = None
        self.gammaD = None
        self.C = None # the surface allocation vector for all events
        self.gammaMF = None
        self.gammaFM = None
        self.etaMF = None
        self.etaFM = None
        self.componentsMF = None
        self.componentsFM = None
        self.weightMF = None
        self.weightFM = None
        self.Z_MF = None # component indicator for MF points (real + ghost)
        self.Z_FM = None # component indicator for MF points (real + ghost)
#        self.Z_FM = None # component indicator for MF process
#        self.Z_0 = None # component indicator for the outside process
        self.alpha_MF = None # DP precision for MF surface mixture
        self.alpha_FM = None # DP precision for FM surface mixture
        
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gammaMF', 'gammaFM', 
                                 'componentsMF', 'weightMF',
                                 'componentsFM', 'weightFM',
                                 'N_MF', 'N_FM', 'C', 'etaMF', 'etaFM',
                                 'alpha_MF', 'alpha_FM',
                                 'Z_MF', 'Z_FM'] # track the Z labels as well!
        # log-likelihood
        #self.log-lik-terms = None # each pair's contribution to the log-likelihood
        self.log_lik = None # total log-likelihood
        # posterior inference (summary statistics and chains)
        self.maxIter = None
        self.burn = 0
        self.thin = 1
        self.chains = {param: list() for param in self.params_to_record}
            # a dictionary for parameter samples
            
    def evalLikelihood(self, subset=None):
        '''
        Evaluate likelihood
        Returns total log likelihood 
        (Currently no implementation on subset!!)
        
        (06/28: not modified yet - not used right now)
        '''

        # Right now: STUPID WAY - sum over individual entries
        # later might change
        
        LLik = np.sum(evalLLikelihood(self.L, self.indsMF, self.indsFM, self.muL, 
                                      self.gammaL, subset=subset, log=True))
        DLik = np.sum(evalDLikelihood(self.D, self.indsMF, self.indsFM, 
                                      self.muD, self.muNegD, 
                                      self.gammaD, subset=subset, log=True))
        
        X = getPoints(self.E)
        N = len(self.E)
        
        MFLik = np.sum(evalDensity(X[self.indsMF,:], self.weightMF, self.components, log=True)) if len(self.indsMF) > 0 else 0
        FMLik = np.sum(evalDensity(X[self.indsFM,:], self.weightFM, self.components, log=True)) if len(self.indsFM) > 0 else 0
        Lik0 =  np.sum(evalDensity(X[self.inds0,:], self.weight0, self.components, log=True)) if len(self.inds0) > 0 else 0 
        
        counts = np.array([len(self.inds0), len(self.indsMF), len(self.indsFM)])
        
        
        total = LLik + DLik + MFLik + FMLik + Lik0 + counts.dot(np.log(self.probs))
        total += N * np.log(self.gamma) - np.log(range(N)).sum() - self.gamma
        
#        if subset is None:
#            to_add = (N_MF * np.log(self.gammaMF) + N_FM * np.log(self.gammaFM) - 
#                      np.log(range(1,N_MF+1)).sum() - np.log(range(1,N_FM+1)).sum())
#            total += to_add - (self.gammaMF + self.gammaFM)
#            self.log_lik = total
            
        return total
    
    def updateTypeIndicator(self):
        '''
        Update the type indicator "C" for each point in the dataset
        Returns a length-N vector of indicators (values in 0, 1, 2, 3)
        
        0, 1 = ghost events on MF, FM surface
        2, 3 = real events on MF, FM surface
        '''

        N = len(self.E)
        indsall = list(range(N))
        
        condProbs = np.empty((N,4))
        
        # pre-compute some stuff
        ## score model density for all the outside events (C=0 and 1)
        log_score0 = (evalLLikelihood(self.L, [], [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], [], self.muD, self.muNegD, self.gammaD))
        
        ## MF surface scale + density
        logMF = (evalDensity(getPoints(self.E), self.weightMF, self.componentsMF) + 
                 np.log(self.gammaMF))
        
        ## FM surface scale + density
        logFM = (evalDensity(getPoints(self.E, flip=True), self.weightFM, self.componentsFM) + 
                 np.log(self.gammaFM))
        
        # C=0 (ghost MF)
        condProbs[:,0] = log_score0 + logMF + np.log(1-self.etaMF)
        
        # C=1 (ghost FM)
        condProbs[:,1] = log_score0 + logFM + np.log(1-self.etaFM)
        
        # C=2 (real MF)
        condProbs[:,2] = (evalLLikelihood(self.L, indsall, [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, indsall, [], self.muD, self.muNegD, self.gammaD) + 
                 logMF + np.log(self.etaMF))
        
        # C=3 (real FM)
        condProbs[:,3] = (evalLLikelihood(self.L, [], indsall, self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], indsall, self.muD, self.muNegD, self.gammaD) + 
                 logFM + np.log(self.etaFM))
        
        self.C = np.apply_along_axis(lambda v: rng.choice(range(4), replace=False, 
                                                      p=getProbVector(v)), 1, condProbs)

        
        return
        

    
    def fit(self, E, L, D, samples = 1000, burn = 0, thin = 1, random_seed = 42, 
            verbose = True, debugHack = False):
        '''
        Fit the model via MCMC
        '''
        # set up
        self.E = E
        self.L = L
        self.D = D
        N = len(E)
        #self.log-lik-terms = np.empty(len(E))
        self.burn = burn
        self.thin = thin
        self.maxIter = samples * thin + burn
        
        np.random.seed(random_seed)
        
        # (Take care of all the gamma draws at the beginning???)
        
        
        # initialize
        # 1) scores
        self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, self.linkInitialThreshold)
        self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds)
        # 2) Gaussian mixture components
        
        ## first: get some initial values of alpha...
        self.alpha_MF, self.alpha_FM = rng.gamma(self.alphaPrior['a'], 1/self.alphaPrior['a'], size=2)
        
        ## right now: only use "real" events to get initial estimates of the GMM stuff
        X = getPoints(self.E)
        X_MF = X[self.indsMF,:]; X_FM = X[self.indsFM,:][:,(1,0)]
        
        ## MF surface
        self.componentsMF, self.Z_MF = initializeDPGMM(X_MF, self.muPrior, 
                                                       self.precisionPrior, self.K, self.Kmax)
        self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
        K_MF = len(np.unique(self.Z_MF))
        self.alpha_MF = updateAlpha(K_MF, N, self.alpha_MF, self.alphaPrior)
        
        ## FM surface
        self.componentsFM, self.Z_FM = initializeDPGMM(X_FM, self.muPrior, self.precisionPrior, 
                                                       self.K, self.Kmax)
        self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
        K_FM = len(np.unique(self.Z_FM))
        self.alpha_FM = updateAlpha(K_FM, N, self.alpha_FM, self.alphaPrior)
        # 3) get C
        ## right now: 
        ## randomly assign all outsiders to either C=1 or C=0 to start with
        self.C = rng.choice(range(2), N)
        self.C[self.indsMF] = 2
        self.C[self.indsFM] = 3
        
        self.inds0MF = np.where(self.C == 0)[0]
        self.inds0FM = np.where(self.C == 1)[0]
        # 4) gamma and eta
        self.gammaMF, self.gammaFM = updateGamma(self.C, self.PPGammaPrior)
        self.etaMF, self.etaFM = updateEta(self.C, self.etaPrior)
        
        if(verbose):
            print('Initialization done!')
        
        # MCMC
        # 05/09 debug: hack it to fix everything else except E_MF, E_FM and see how it goes...
        for it in range(self.maxIter):
            ## 1. the score models
            # HACK it for debugging purposes:
            if debugHack:
                self.muL, self.gammaL = Settings['muL'], Settings['gammaL']
                self.muD, self.muNegD, self.gammaD = Settings['muD'], Settings['muNegD'], Settings['gammaD']
            else:
                self.muL, self.gammaL = updateLModel(self.L, self.indsMF, self.indsFM, self.muL, 
                                                     self.gammaL, self.ScoreGammaPrior)
                
                self.muD, self.muNegD, self.gammaD = updateDModel(self.D, self.indsMF, self.indsFM, 
                                                                  self.muD, self.muNegD, 
                                                                  self.gammaD, self.ScoreGammaPrior)                

            
            ## 2. the point configurations
            
            ## 2.1 update event type allocation
            self.updateTypeIndicator()
            
            ## 2.2 bookkeeping
            self.indsMF = np.where(self.C == 2)[0]
            self.indsFM = np.where(self.C == 3)[0]
            self.inds0MF = np.where(self.C == 0)[0]
            self.inds0FM = np.where(self.C == 1)[0]
            
               
            ## 3. Update gamma and eta
            self.gammaMF, self.gammaFM = updateGamma(self.C, self.PPGammaPrior)
            self.etaMF, self.etaFM = updateEta(self.C, self.etaPrior)
            
            ## 4. Update the DP Gaussian Mixture Model for the densities
            # 4.1 MF surface
            MF_indsall = list(self.indsMF) + list(self.inds0MF)
            X_MF = X[MF_indsall,:]
            self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
            self.componentsMF = updateGaussianComponents(X_MF, self.Z_MF, self.componentsMF,
                                                         self.muPrior, self.precisionPrior)
            self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
            K_MF = len(np.unique(self.Z_MF))
            self.alpha_MF = updateAlpha(K_MF, N, self.alpha_MF, self.alphaPrior)
            
            # 4.2 FM surface
            FM_indsall = list(self.indsFM) + list(self.inds0FM)
            X_FM = X[FM_indsall,:][:,(1,0)]
            self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
            self.componentsFM = updateGaussianComponents(X_FM, self.Z_FM, self.componentsFM,
                                                         self.muPrior, self.precisionPrior)
            self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
            K_FM = len(np.unique(self.Z_FM))
            self.alpha_FM = updateAlpha(K_FM, N, self.alpha_FM, self.alphaPrior)
            
            if verbose and it<burn:
                print('Burn-in at iteration {}/{}.'.format(it, self.maxIter))
           
            
            ## 5. Save parameter in chains if...
            if (it >= burn) & ((it+1-burn) % thin == 0):
                self.chains['muL'].append(self.muL)
                self.chains['muD'].append(self.muD)
                self.chains['muNegD'].append(self.muNegD)
                self.chains['gammaL'].append(self.gammaL)
                self.chains['gammaD'].append(self.gammaD)
                self.chains['N_MF'].append(len(self.indsMF))
                self.chains['N_FM'].append(len(self.indsFM))
                self.chains['gammaMF'].append(self.gammaMF)
                self.chains['gammaFM'].append(self.gammaFM)
                self.chains['etaMF'].append(self.etaMF)
                self.chains['etaFM'].append(self.etaFM)
                self.chains['C'].append(self.C)
                self.chains['componentsMF'].append(self.componentsMF)
                self.chains['componentsFM'].append(self.componentsFM)
                self.chains['weightMF'].append(self.weightMF)
                self.chains['weightFM'].append(self.weightFM)
                self.chains['alpha_MF'].append(self.alpha_MF)
                self.chains['alpha_FM'].append(self.alpha_FM)
                
                ## also tracking Z_MF's and Z_FM's
                self.chains['Z_MF'].append(self.Z_MF)
                self.chains['Z_FM'].append(self.Z_FM)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}.'.format(it, self.maxIter))
            
        return
    
    def plotChains(self, param, s=None, savepath=None):
        '''
        param: parameter name
        s: which iteration to display
        savepath: path to save figure
        '''
        if param.startswith('compo'):
            
            chain = self.chains[param]
            if s >= len(chain) or s is None:
                s = -1
            components = chain[s]
                
            # also load the mixture weights
            suffix = param[-2:]
            weights = self.chains['weight'+suffix][s]
            
            # get the relevant data
            C = self.chains['C'][s]
            
            if suffix=='MF':
                data = getPoints(self.E)[C%2==0,:]
            else:
                data = getPoints(self.E)[C%2==1,:][:,(1,0)]
            
            # get the min, max range
            Amin = np.min(data); Amax = np.max(data)
            
            # make density contour plot
            #Amin = 15.0; Amax = 50.0
            x = np.linspace(Amin, Amax)
            y = np.linspace(Amin, Amax)
            X, Y = np.meshgrid(x, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            
            Z = evalDensity(XX, weights, components, log=True)
            Z = Z.reshape(X.shape)
            
            plt.contourf(X,Y,Z)
            
            # overlay with the predicted age-pair points
            
            plt.scatter(data[:,0], data[:,1], c="black")
            
            
            plt.title('predicted log-density of the {} surface'.format(suffix))
            plt.xlabel('transmitter age')
            plt.ylabel('recipient age')
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
 
        elif param=="C":
            # s: can serve as the starting point for querying the chain
            
            def tabulate(C):
                counts = np.empty(shape=4)
                for k in range(4):
                    counts[k] = np.sum(C==k)
                return counts
            
            if s is None or s<0 or s>=len(self.chain['C']):
                s = 0
                
            Cs = np.array(self.chains['C'][s:])
            all_counts = np.apply_along_axis(tabulate, 1, Cs)
            Counts_mean = np.mean(all_counts,axis=0)
            Counts_std = np.std(all_counts,axis=0)
            
            ind = np.arange(len(Counts_mean))
            plt.bar(ind, Counts_mean, 0.5, yerr = Counts_std,
                    error_kw=dict(lw=3, capsize=3, capthick=2))
            plt.title('Number of points allocated to each type through the chain')
            plt.xticks(ind, ('MF0', 'FM0', 'MF', 'FM'))
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
            
        elif param.startswith('weight'):
            chain = np.array(self.chains[param])
            for k in range(self.Kmax):
                #this_label = 'comp '+str(k)
                this_label = str(k)
                plt.plot(chain[:,k],"-",label=this_label)
            #plt.legend(loc='upper right')
            plt.xlabel('Samples')
            plt.title('Traceplot of {}'.format(param))
            plt.legend()
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
# =============================================================================
#         elif param.startswith('prob'):
#             chain = np.array(self.chains[param])
#             for h in range(3):
#                 plt.plot(chain[:,h],"-",label=str(h))
#             plt.legend('upper right')
#             plt.show()
# =============================================================================
        else:
            plt.plot(self.chains[param])
            if savepath is not None:
                plt.savefig(savepath)
            
            plt.title('Traceplot of {}'.format(param))
            plt.xlabel('Samples')
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
        return



 
#%%
# try running the updated new model

# OUTDATED version of precision matrices

#Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
#      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
#      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
#      "weight": np.ones(3), "probs": np.ones(3),
#      "gammaPP": {'n0': 1, 'b0': 0.02},
#      "eta": {'a': 1, 'b': 1}}  
#
#model = LatentPoissonGMM2(Priors = Pr, K=3)

#%%
# Aug 29, 2020
# try running the updated GP-GMM model
Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
      "probs": np.ones(3), 
      "alpha": {'a': 2.0, 'b':3.0},
      "gammaPP": {'n0': 1, 'b0': 0.02},
      "eta": {'a': 1, 'b': 1}}

model = LatentPoissonDPGMM2(Priors = Pr, K=3, Kmax=10)

## Some completely made-up data that won't follow the model at all
#E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
#L = (1-0.3)* np.random.random_sample(100) + 0.3
#D = np.random.random_sample(100)
#
#model.fit(E,L,D, samples=2000, burn=0, random_seed = 71)
# for this one: one of the event sets will eventually get empty...

#%%

# =============================================================================
# V1
## adapted from the 2-surface version
#Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
#            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
#            'gammaL': 1, 'gammaD': 1, 
#            'weightMF': np.array([0.4, 0.3, 0.3]), 'weightFM': np.array([0.4, 0.3, 0.3]),
#            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
#                             ([40,25], np.diag([1/4,1/9]))],
#             'componentsFM': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
#                              ([25,40], np.diag([1/9,1/4]))]}
              ## In this setting: components of MF and FM are essentially the same!
              
# =============================================================================

# =============================================================================
# # V2
# ## Try another setting
# ## MF and FM GMM different weights but same components
#Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
#            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
#            'gammaL': 1, 'gammaD': 1, 
#            'weightMF': np.array([0.8, 0.1, 0.1]), 'weightFM': np.array([0.1, 0.8, 0.1]),
#            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
#                             ([40,25], np.diag([1/4,1/9]))],
#            'componentsFM': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
#                             ([25,40], np.diag([1/9,1/4]))]}
# =============================================================================

# V3
# Try yet another setting
# MF and FM GMM different components too
Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.6, 0.2, 0.2]), 'weightFM': np.array([0.6, 0.2, 0.2]),
            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([30,20], np.diag([1/9,1/9])), 
                             ([25,25], np.diag([1/4,1/9]))],
            'componentsFM': [([20,40], np.diag([1/4,1/4])), ([25,45], np.diag([1/9,1/9])), 
                             ([25,40], np.diag([1/9,1/4]))]}

            

#E, L, D = simulateLatentPoissonGMM2(Settings)
#
#E_MF = {i:a for i,a in E.items() if i in range(80)}
#E_FM = {i:a[::-1] for i,a in E.items() if i in range(80,150)}
#
#
## visualize a bit
#X = getPoints(E)
#plt.plot(X[:,0], X[:,1], "o")
#plt.show()
#
#plt.plot(L,"o")
#plt.show()
#
#plt.plot(D, "o")
#plt.show()


#%%

# 08/29/20 PM change:
# use covariance matrix instead of precision matrix

Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001,
                'covariance': np.eye(2)*10000},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2), 'Scale': np.eye(1)},
      "probs": np.ones(3), 
      "gammaPP": {'n0': 1, 'b0': 0.02},
      "eta": {'a': 1, 'b': 1},
      "alpha": {'a': 2.0, 'b':3.0}}

model = LatentPoissonDPGMM2(Priors = Pr, K=3, Kmax=10)


# # V2
# ## Try another setting
# ## MF and FM GMM different weights but same components
#Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
#            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
#            'gammaL': 1, 'gammaD': 1, 
#            'weightMF': np.array([0.8, 0.1, 0.1]), 'weightFM': np.array([0.1, 0.8, 0.1]),
#            'componentsMF': [([40,40], np.diag([4,4])), ([25,25], np.diag([9,9])), 
#                             ([40,25], np.diag([4,9]))],
#            'componentsFM': [([40,40], np.diag([4,4])), ([25,25], np.diag([9,9])), 
#                             ([25,40], np.diag([9,4]))]}
            

# V3
# Try yet another setting
# MF and FM GMM different components too
Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.6, 0.2, 0.2]), 'weightFM': np.array([0.6, 0.2, 0.2]),
            'componentsMF': [([40,40], np.diag([4,4])), ([30,20], np.diag([9,9])), 
                             ([25,25], np.diag([4,9]))],
            'componentsFM': [([20,40], np.diag([4,4])), ([25,45], np.diag([9,9])), 
                             ([25,40], np.diag([9,4]))]}


# V4
# increase number of points
Settings = {'N_MF': 160, 'N_FM': 120, 'N_MF0':40, 'N_FM0': 30, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.5, 0.4, 0.1]), 'weightFM': np.array([0.6, 0.2, 0.2]),
            'componentsMF': [([40,40], np.diag([4,4])), ([30,20], np.diag([9,9])), 
                             ([25,25], np.diag([4,9]))],
            'componentsFM': [([20,40], np.diag([4,4])), ([30,45], np.diag([9,9])), 
                             ([25,30], np.diag([9,4]))]}


#%%
E, L, D = simulateLatentPoissonGMM2(Settings)

E_MF = {i:a for i,a in E.items() if i in range(160)}
E_FM = {i:a[::-1] for i,a in E.items() if i in range(160,280)}


# visualize a bit
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show()

X_MF = getPoints(E_MF)
plt.plot(X_MF[:,0], X_MF[:,1], "o")
plt.xlim((12,45))
plt.ylim((12,45))
plt.title('MF surface ground truth')
plt.show()

X_FM = getPoints(E_FM)
plt.plot(X_FM[:,0], X_FM[:,1], "o")
plt.xlim((15,50))
plt.ylim((15,50))
plt.title('FM surface ground truth')
plt.show()

plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

#%%
# try to fit 
model.fit(E, L, D, samples=3000, burn=0, random_seed = 89, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF',savepath='N_MF_diffGMM.pdf')
model.plotChains('N_FM',savepath='N_FM_diffGMM.pdf')
model.plotChains('weightMF',savepath='weightMF_diffGMM.pdf')
model.plotChains('weightFM',savepath='weightFM_diffGMM.pdf')
model.plotChains('etaMF')
model.plotChains('gammaMF')
model.plotChains('muL')
model.plotChains('muD')
model.plotChains('muNegD')
model.plotChains('alpha_MF')
model.plotChains('alpha_FM')

### Finding based on v1 and v2
## The MF and FM surfaces should have different patterns 
## (If pattern similar, then a lot of points will be dragged to one surface and the chain breaks down)
## the GMM may experience label switching problems
## but otherwise not too bad (the type indicators not exactly correct, but mostly alright)

### Based on V3
### As long as the components are somewhat different, it works!

model.plotChains('componentsFM', s=2800, savepath='FM_surface.pdf')
model.plotChains('componentsMF', s=2800, savepath='MF_surface.pdf')

# save model to an object
# pkl.dump(model, file=open("Aug29_V4_3000iters.pkl",'wb'))

pkl.dump(model, file=open("Sept24_V4_3000iters.pkl",'wb'))

#%%
# load a previous model for plotting...
#model = pkl.load(open("Aug29_V4_3000iters.pkl",'rb'))


#%%

# Oct 10, 2020
# try it on real data

import pandas as pd

dat = pd.read_csv("../200928_data_not_unlike_real_data.csv")

# filter out some "low linked prob" data points

# a heuristic fix: only keep those not very close to 0 or 1

dat = dat[(dat.POSTERIOR_SCORE_LINKED > 0.2) & (dat.POSTERIOR_SCORE_LINKED < 0.98) &
          (dat.POSTERIOR_SCORE_MF > 0.02) & (dat.POSTERIOR_SCORE_MF < 0.98)]

L = np.array(dat.POSTERIOR_SCORE_LINKED)
D = np.array(dat.POSTERIOR_SCORE_MF)

edges = np.array(dat[['MALE_AGE_AT_MID','FEMALE_AGE_AT_MID']])
nr = edges.shape[0]

E = dict(zip(range(nr), edges))

plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

#%%
# try fitting the model on this dataset

model.fit(E, L, D, samples=3000, burn=0, random_seed = 89, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF',savepath='N_MF_diffGMM.pdf')
model.plotChains('N_FM',savepath='N_FM_diffGMM.pdf')
model.plotChains('weightMF',savepath='weightMF_diffGMM.pdf')
model.plotChains('weightFM',savepath='weightFM_diffGMM.pdf')
model.plotChains('etaMF')
model.plotChains('gammaMF')
model.plotChains('muL')
model.plotChains('muD')
model.plotChains('muNegD')
model.plotChains('alpha_MF')
model.plotChains('alpha_FM')

model.plotChains('componentsFM', s=2800, savepath='FM_surface.pdf')
model.plotChains('componentsMF', s=2800, savepath='MF_surface.pdf')

model.plotChains('componentsFM', s=2250, savepath='FM_surface.pdf')
model.plotChains('componentsMF', s=2250, savepath='MF_surface.pdf')


pkl.dump(model, file=open("Oct10_synData_3000iters.pkl",'wb'))