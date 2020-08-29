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
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

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
                                 'N_MF', 'N_FM', 'C', 'etaMF', 'etaFM']
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
        
        self.C = np.apply_along_axis(lambda v: choice(range(4), replace=False, 
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
        # 2) Gaussian components
        ## right now: only use "real" events to get initial estimates of the GMM stuff
        X = getPoints(self.E)
        X_MF = X[self.indsMF,:]; X_FM = X[self.indsFM,:][:,(1,0)]
        self.componentsMF, self.Z_MF = initializeGMM(X_MF, self.K)
        self.weightMF = updateMixtureWeight(self.Z_MF, self.weightPrior)
        self.componentsFM, self.Z_FM = initializeGMM(X_FM, self.K)
        self.weightFM = updateMixtureWeight(self.Z_FM, self.weightPrior)
        # 3) get C
        ## right now: 
        ## randomly assign all outsiders to either C=1 or C=0 to start with
        self.C = choice(range(2), N)
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
            
            ## 4. Update the Gaussian Mixture Model for the densities
            # 4.1 MF surface
            MF_indsall = list(self.indsMF) + list(self.inds0MF)
            X_MF = X[MF_indsall,:]
            self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
            self.weightMF = updateMixtureWeight(self.Z_MF, self.weightPrior)
            self.componentsMF = updateGaussianComponents(X_MF, self.Z_MF, self.componentsMF,
                                                         self.muPrior, self.precisionPrior)
            # 4.2 FM surface
            FM_indsall = list(self.indsFM) + list(self.inds0FM)
            X_FM = X[FM_indsall,:][:,(1,0)]
            self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
            self.weightFM = updateMixtureWeight(self.Z_FM, self.weightPrior)
            self.componentsFM = updateGaussianComponents(X_FM, self.Z_FM, self.componentsFM,
                                                         self.muPrior, self.precisionPrior)

            
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
                
                if verbose:
                    print('Parameters saved at iteration {}/{}.'.format(it, self.maxIter))
            
        return
    
    def plotChains(self, param, savepath=None):
        if param.startswith('compo'):
            # don't deal with components right now...
            pass
        elif param=="C":
            # don't deal with C indicators either...
            pass
        elif param.startswith('weight'):
            chain = np.array(self.chains[param])
            for k in range(self.K):
                #this_label = 'comp '+str(k)
                this_label = str(k)
                plt.plot(chain[:,k],"-",label=this_label)
            #plt.legend(loc='upper right')
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
            plt.show()
            
        return



 
#%%
# try running the updated new model
 
Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
      "weight": np.ones(3), "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02},
      "eta": {'a': 1, 'b': 1}}  

model = LatentPoissonGMM2(Priors = Pr, K=3)

## Some completely made-up data that won't follow the model at all
#E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
#L = (1-0.3)* np.random.random_sample(100) + 0.3
#D = np.random.random_sample(100)
#
#model.fit(E,L,D, samples=2000, burn=0, random_seed = 71)
# for this one: one of the event sets will eventually get empty...


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
Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.8, 0.1, 0.1]), 'weightFM': np.array([0.1, 0.8, 0.1]),
            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([40,25], np.diag([1/4,1/9]))],
            'componentsFM': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([25,40], np.diag([1/9,1/4]))]}
# =============================================================================

# V3
## Try yet another setting
## MF and FM GMM different components too
#Settings = {'N_MF': 80, 'N_FM': 70, 'N_MF0':20, 'N_FM0': 30, 
#            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
#            'gammaL': 1, 'gammaD': 1, 
#            'weightMF': np.array([0.6, 0.2, 0.2]), 'weightFM': np.array([0.6, 0.2, 0.2]),
#            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([30,20], np.diag([1/9,1/9])), 
#                             ([25,25], np.diag([1/4,1/9]))],
#            'componentsFM': [([20,40], np.diag([1/4,1/4])), ([25,45], np.diag([1/9,1/9])), 
#                             ([25,40], np.diag([1/9,1/4]))]}

            

E, L, D = simulateLatentPoissonGMM2(Settings)

E_MF = {i:a for i,a in E.items() if i in range(80)}
E_FM = {i:a[::-1] for i,a in E.items() if i in range(80,150)}


# visualize a bit
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show()

plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

#%%
# try to fit 
model.fit(E, L, D, samples=2000, burn=0, random_seed = 89, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF')
model.plotChains('N_FM')
model.plotChains('weightMF','weightMF_diffGMM.pdf')
model.plotChains('weightFM','weightFM_diffGMM.pdf')
model.plotChains('etaMF')
model.plotChains('muL')
model.plotChains('muD')

### Finding based on v1 and v2
## The MF and FM surfaces should have different patterns 
## (If pattern similar, then a lot of points will be dragged to one surface and the chain breaks down)
## the GMM may experience label switching problems
## but otherwise not too bad (the type indicators not exactly correct, but mostly alright)

### Based on V3
### As long as the components are somewhat different, it works!