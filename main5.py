#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:35:05 2020

@author: fan

adapt "main3" to get a less flexible version

further adapt "main4" to fix real data experiment issues
"""

#%%

# implement the Dirichlet process version of 3-surface model
import os
#os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

# 10/11/2020: changed working directory
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

#%%
from utilsHupdate import *


## 01/09/2021: a version with 3 separate surfaces (separate components and weights)

## 09/05/2021: a new version with separate priors for gammaL and gammaD (for more control)

## 09/15/2021: try to put fixing muD and muNegD inside the update utils (and fix the D_centers entry order)

## 09/15/2021 UPDATE (again):
## treat extreme scores (0 or 1 in L or D) more carefully
## (1) allocate L = 1 pairs all to the MF and FM surfaces (exclude them from the score model est.)
## (2) allocate D = 0 pairs all to MF, and D = 1 pairs all to FM (exclude them from the score model est.)
## (3) DO include those pairs in the spatial model


# just make sure the getProbVector function works properly!
def getProbVector(p):
    
    # some "anti-infinity" truncation to address numerical issues
    
    p[p==np.inf] = 3000
    p[p==-np.inf] = -3000
    
    p = np.exp(p - np.max(p))
    
    #print(p)
    
    return p/p.sum()


class LatentPoissonDPHGMM:
    def __init__(self, Priors, K=3, Kmax = 10, linkThreshold=0.6):
        '''
        Inialize an instance of the LatentPoisson Hierarchical GMM model;
        Priors: a big dictionary of all the priors
            - "gammaPP": prior dictionary for the NHPP scale (gamma); 
                need "n0" and "b0"
            - "probs": prior vector (length 3) for the surface probability/proportion vector
            - "muGMM": prior dictionary for the means in Gaussian Mixture; 
                need "mean" and "precision"
            - "precisionGMM": prior dictionary for the precision matrices in Gaussian Mixture;
                need "df" and "invScale"
            - "weight": prior vector (length K) for Gaussian Mixture weight;
            - "gammaL": prior dictionary for inverse variance of the linkage (K) score;
                need "nu0" and "sigma0"
            - "gammaD": prior dictionary for inverse variance of the linkage (K) score;
                need "nu0" and "sigma0"
            - "alpha": prior for the DP precision; need "a" and "b"
        '''
        self.name = "Latent Poisson Process with Dirichlet Process Gaussian Mixture density"
        self.linkInitialThreshold = linkThreshold
        # number of mixture components
        self.K = K
        self.Kmax = Kmax
        # prior part
        #self.ScoreGammaPrior = Priors["gammaScore"]
        self.GammaLPrior = Priors["gammaL"]
        self.GammaDPrior = Priors["gammaD"]
        self.muPrior = Priors["muGMM"]
        self.precisionPrior = Priors["precisionGMM"]
        #self.weightPrior = Priors["weight"]
        self.PPGammaPrior = Priors["gammaPP"]
        self.probPrior = Priors["probs"]
        self.alphaPrior = Priors['alpha']
        # data part
        self.E = None # all the (a_M,a_F) pairs
        self.L = None # all the linked scores
        self.D = None # all the direction scores
        self.indsMF = None # indices on the MF surface
        self.indsFM = None # indices on the FM surface
        self.inds0 = None # indices for the outsider points
        ## 09/15/2021: indices that should not be included in L or D models
        self.indsNoD = None
        self.indsNoL = None
        ## AND indices aht should rely on L or D models only for sampling indicator
        ## 09/20/2021: adjust the LModelOnly to potential MF and potential FM's 
        ##             and keep DModelOnly (but not use it) for now
        #self.LModelOnly = None
        self.potMF = None
        self.potFM = None
        self.DModelOnly = None
        #self.E_MF = None # event set on MF surface
        #self.E_FM = None # event set on FM surface
        #self.E_0 = None # event set on the outside
        # parameters
        self.muL = None
        self.muD = None
        self.muNegD = None
        self.gammaL = None
        self.gammaD = None
        self.gamma = None # the scale parameter for the entire NHPP
        self.probs = None # the surface probability/proportions vector
        self.C = None # the surface allocation vector for all events
        self.componentsMF = None # components for MF
        self.componentsFM = None # components for FM
        self.components0 = None # components for outside surface
        self.weightMF = None
        self.weightFM = None
        self.weight0 = None # GMM weights for the "outside" surface
        self.Z_MF = None # component indicator for MF process
        self.Z_FM = None # component indicator for MF process
        self.Z_0 = None # component indicator for the outside process
        self.alpha_MF = None # DP precision for the MF mixture components
        self.alpha_FM = None # DP precision for the FM mixture components
        self.alpha_0 = None # DP precision for the outside mixture components
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gamma', 'probs', 'C', 
                                 'componentsMF', 'componentsFM', 'components0', 
                                 'weightMF', 'weightFM', 'weight0',
                                 'alpha_MF', 'alpha_FM', 'alpha_0',
                                 'L', 'D']
        # log-likelihood
        #self.log-lik-terms = None # each pair's contribution to the log-likelihood
        self.log_lik = None # total log-likelihood
        # posterior inference (summary statistics and chains)
        self.maxIter = None
        self.burn = 0
        self.thin = 1
        self.chains = {param: list() for param in self.params_to_record}
            # a dictionary for parameter samples
        # 10/26/2020: also record the the likelihood in each iteration 
        # (in order to extract MAP estimate)
        self.chains['loglik'] = list()
            
    def evalLikelihood(self, subset=None):
        '''
        Evaluate likelihood
        Returns total log likelihood 
        (Currently no implementation on subset!!)
        '''

        # Right now: STUPID WAY - sum over individual entries
        # later might change
        
        ## 09/15/2021: update likelihood eval (exclude points with extreme L or D values)
        LLik = np.sum(evalLLikelihood(self.L, self.indsMF, self.indsFM, self.muL, 
                                      self.gammaL, indsNoL = self.indsNoL, 
                                      subset=subset, log=True))
        DLik = np.sum(evalDLikelihood(self.D, self.indsMF, self.indsFM, 
                                      self.muD, self.muNegD, 
                                      self.gammaD, indsNoD = self.indsNoD,
                                      subset=subset, log=True))
        
        X = getPoints(self.E)
        N = len(self.E)
        
        MFLik = np.sum(evalDensity(X[self.indsMF,:], self.weightMF, self.componentsMF, log=True)) if len(self.indsMF) > 0 else 0
        FMLik = np.sum(evalDensity(X[self.indsFM,:], self.weightFM, self.componentsFM, log=True)) if len(self.indsFM) > 0 else 0
        Lik0 =  np.sum(evalDensity(X[self.inds0,:], self.weight0, self.components0, log=True)) if len(self.inds0) > 0 else 0 
        
        counts = np.array([len(self.inds0), len(self.indsMF), len(self.indsFM)])
        
        
        total = LLik + DLik + MFLik + FMLik + Lik0 + counts.dot(np.log(self.probs))
        total += N * np.log(self.gamma) - np.log(range(1,N+1)).sum() - self.gamma
        
#        if subset is None:
#            to_add = (N_MF * np.log(self.gammaMF) + N_FM * np.log(self.gammaFM) - 
#                      np.log(range(1,N_MF+1)).sum() - np.log(range(1,N_FM+1)).sum())
#            total += to_add - (self.gammaMF + self.gammaFM)
#            self.log_lik = total
            
        return total
    
    def updateTypeIndicator(self):
        '''
        Update the type indicator "C" for each point in the dataset
        Returns a length-N vector of indicators (values in 0, 1, 2)
        '''
        
        ## 09/15/2021 update: add in indices for those points with extreme D or L scores
        ## (plug in 0 as their log lik. entries)

        N = len(self.E)
        indsall = list(range(N))
        
        condProbs = np.empty((N,3))
        
        # h=0 (all outside)
        condProbs[:,0] = (evalLLikelihood(self.L, [], [], self.muL, 
                 self.gammaL, indsNoL = self.indsNoL) + 
                 evalDLikelihood(self.D, [], [], self.muD, self.muNegD, 
                                 self.gammaD, indsNoD = self.indsNoD) + 
                 evalDensity(getPoints(self.E), self.weight0, self.components0) +
                 np.log(self.probs[0]))
        
        # h=1 (all in MF)
        condProbs[:,1] = (evalLLikelihood(self.L, indsall, [], self.muL, 
                 self.gammaL, indsNoL = self.indsNoL) + 
                 evalDLikelihood(self.D, indsall, [], self.muD, self.muNegD, 
                                 self.gammaD, indsNoD = self.indsNoD) + 
                 evalDensity(getPoints(self.E), self.weightMF, self.componentsMF) +
                 np.log(self.probs[1]))
        
        # h=2 (all in FM)
        condProbs[:,2] = (evalLLikelihood(self.L, [], indsall, self.muL, 
                 self.gammaL, indsNoL = self.indsNoL) + 
                 evalDLikelihood(self.D, [], indsall, self.muD, self.muNegD, 
                                 self.gammaD, indsNoD = self.indsNoD) + 
                 evalDensity(getPoints(self.E), self.weightFM, self.componentsFM) +
                 np.log(self.probs[2]))
        
        self.C = np.apply_along_axis(lambda v: choice(range(3), replace=False, 
                                                      p=getProbVector(v)), 1, condProbs)

        
        return
        

    
    def fit(self, E, L, D, samples = 1000, burn = 0, thin = 1, random_seed = 42, 
            verbose = True, debugHack = False, 
            fixed_alloc = False, thresholds = [0.6,0.5], D_centers = [0.5, -0.5],
            def_event_inds = [None, None], extreme_inds = [None, None], 
            L_D_model_inds = [None, None, None]):
        '''
        Fit the model via MCMC
        fixed_alloc: whether or not using fixed allocation with thresholds
        thresholds: [threshold for L score, threshold for D score (> ... --> MF)]
        D_centers: [muD, muNegD] - fixed mixture centers for easier stability
        def_event_inds: [def_MF, def_FM] - inds for points definitely inside MF / FM surface
        extreme_inds: [ext_L, ext_D] - inds with extreme L or D scores
        L_D_only_inds: [L_model_only, D_model_only] - inds that only depend on L or D models for sampling C
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
        
        ## 09/15/2021: read in the indices for extreme L or D scores
        self.indsNoL, self.indsNoD = extreme_inds
        ## 09/20/2021 adjustment here
        self.potMF, self.potFM, self.DModelOnly = L_D_model_inds
        def_MF, def_FM = def_event_inds
        
        # (Take care of all the gamma draws at the beginning???)
        
        
        # initialize
        # 1) scores
        ## 01/06/2021 change: add "fixed allocation" option
        if fixed_alloc:
            thresL, thresD = thresholds
            self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, thresL, indsNoL = self.indsNoL)
            self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds, 
                                                                                                         thresD, indsNoD = self.indsNoD)
        else:
            self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, self.linkInitialThreshold, 
                                                                        indsNoL = self.indsNoL)
            self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds,
                                                                                                         indsNoD = self.indsNoD)
            
            ## 09/15/2021 update
            ## fix muD and muNegD if D_centers are specified
            if D_centers is not None:
                self.muD, self.muNegD = D_centers
                
        ## 09/15/2021
        ## combine indsMF & indsFM with def_MF & def_FM
        if def_MF is not None and def_FM is not None:
            self.indsMF = list(set(self.indsMF) | set(def_MF))
            self.indsFM = list(set(self.indsFM) | set(def_FM))            
            
        # 2) the PP
        self.gamma, self.probs = initializePP(self.E, self.indsMF, self.indsFM)
        self.C = np.zeros(N)
        self.C[self.indsMF] = 1
        self.C[self.indsFM] = 2
        # 3) DP Gaussian components and weights
        
        ## 01/06/2021 change: 3 surfaces are treated separately (nothing is shared any more)
        ## get points 
        X = getPoints(self.E)
        
        ## some initial estimates of the alphas
        self.alpha_MF, self.alpha_FM, self.alpha_0 = rng.gamma(self.alphaPrior['a'], 1/self.alphaPrior['b'], size=3)
        
        # 3.1) MF surface
        X_MF = X[self.indsMF,:]
        ## components and indicators
        self.componentsMF, self.Z_MF = initializeDPGMM(X_MF, self.muPrior, 
                                                       self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
        
        ## update alpha
        K_MF = len(np.unique(self.Z_MF))
        N_MF = len(self.indsMF)
        self.alpha_MF = updateAlpha(K_MF, N_MF, self.alpha_MF, self.alphaPrior)
        
        # 3.2) FM surface
        X_FM = X[self.indsFM,:]
        ## components and indicators
        self.componentsFM, self.Z_FM = initializeDPGMM(X_FM, self.muPrior, 
                                                       self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
        
        ## update alpha
        K_FM = len(np.unique(self.Z_FM))
        N_FM = len(self.indsFM)
        self.alpha_FM = updateAlpha(K_FM, N_FM, self.alpha_FM, self.alphaPrior)
        
        # 3.1) outside surface
        self.inds0 = np.where(self.C == 0)[0]
        X_0 = X[self.inds0,:]
        ## components and indicators
        self.components0, self.Z_0 = initializeDPGMM(X_0, self.muPrior, 
                                                     self.precisionPrior, self.K, self.Kmax)
        ## GMM weights
        self.weight0 = updateMixtureWeight(self.Z_0, self.alpha_0, self.Kmax)
        
        ## update alpha
        K_0 = len(np.unique(self.Z_0))
        N_0 = len(self.inds0)
        self.alpha_FM = updateAlpha(K_0, N_0, self.alpha_0, self.alphaPrior)
        
        
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
                ## 01/09/2021: fixed_alloc or fixed centers
                self.muL, self.gammaL = updateLModel(self.L, self.indsMF, self.indsFM, self.muL, 
                                                     self.gammaL, self.GammaLPrior, indsNoL = self.indsNoL)
                
            # 01/09/2021: deal with stuff separately based on settings
                
            if fixed_alloc:
                # the inds and C are all fixed throughout, no need to update
                self.muD, self.muNegD, self.gammaD = updateDModel(self.D, self.indsMF, self.indsFM, 
                                                                  self.muD, self.muNegD, 
                                                                  self.gammaD, self.GammaDPrior, 
                                                                  indsNoD = self.indsNoD)
            else:
                # 09/15/2021 udpate
                # fix the two centers for the D scores if specified
                fix = (D_centers is not None)
                self.muD, self.muNegD, self.gammaD = updateDModel(self.D, self.indsMF, self.indsFM, 
                                                                  self.muD, self.muNegD, 
                                                                  self.gammaD, self.GammaDPrior,
                                                                  fixmu = fix, indsNoD = self.indsNoD)
                
                #self.muD, self.muNegD = D_centers
                
                # 01/09/2021: try fixing muL too...
                # self.muL = 2.0
                    
                # 10/26/2020 try stuff
                # 1) fix the value of muD and muNegD to salvage things
                #self.muD = 0.5 # approximate thres = 0.622
                #self.muNegD = -0.5
                
                # 2) truncate muD and muNegD at a non-zero value
                # to stablize things
                #delta = 0.1
                #self.muD = self.muD if self.muD > delta else delta
                #self.muNegD = self.muNegD if self.muNegD < -delta else -delta
            
                # also make sure they are not too positive or negative
                #delta2 = 1.0
                #self.muD = self.muD if self.muD <= delta2 else delta2
                #self.muNegD = self.muNegD if self.muNegD >= -delta2 else -delta2
                    
                    
                ## 2. the point configurations
        
                ## 2.1 update event type allocation
                self.updateTypeIndicator()
                
                ## 2.1.A make adjustments (allocate all definitive points to their surfaces now)
                ## 09/15/2021
                # make sure def_MF and def_FM pairs are included in indsMF and indsFM
                if def_MF is not None and def_FM is not None:
                    self.C[def_MF] = 1
                    self.C[def_FM] = 2
                    
                ## 09/20/2021 adjustment
                if potFM is not None and potFM is not None:
                    not0 = set(np.nonzero(self.C)[0])
                    # assign D = 0 and surface not 0 pairs to FM surface
                    this_FM = list(not0 & set(self.potFM))
                    self.C[this_FM] = 2
                    # and similarly for D = 1 pairs
                    this_MF = list(not0 & set(self.potMF))
                    self.C[this_MF] = 1
        
                ## 2.2 update probs
                self.probs = updateProbs(self.C, self.probPrior)
        
                ## 2.3 bookkeeping
                self.indsMF = np.where(self.C == 1)[0]
                self.indsFM = np.where(self.C == 2)[0]
                self.inds0 = np.where(self.C == 0)[0]
                
                ## combine indsMF & indsFM with def_MF & def_FM
                #self.indsMF = list(set(self.indsMF) | set(def_MF))
                #self.indsFM = list(set(self.indsFM) | set(def_FM))
        
                #self.E_MF = {pair: age for pair, age in self.E.items() if pair in self.indsMF}
                #self.E_FM = {pair: age for pair, age in self.E.items() if pair in self.indsFM}
                #self.E_0 = {pair: age for pair, age in self.E.items() if pair in inds0}
            
        
            ## 3. Update gamma
            self.gamma = np.random.gamma(self.PPGammaPrior['n0']+N, 1/(self.PPGammaPrior['b0']+1))
            
            ## 4. Update the DP Gaussian Mixture Model for the densities
            ## 01/09/2021 change: 3 surfaces separately!
            ### 4.1 MF surface
            if len(self.indsMF) > 0:
                # get surface points
                X_MF = X[self.indsMF,:]
                # update component indicators
                self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
                # re-order labels!
                self.Z_MF, self.componentsMF = relabel(self.Z_MF, self.componentsMF, self.Kmax)
                # update components
                self.componentsMF = updateGaussianComponents(X_MF, self.Z_MF, self.componentsMF, 
                                                             self.muPrior, self.precisionPrior)
                # update weights
                self.weightMF = updateMixtureWeight(self.Z_MF, self.alpha_MF, self.Kmax)
                # update alpha
                K_MF = len(np.unique(self.Z_MF))
                N_MF = len(self.indsMF)
                self.alpha_MF = updateAlpha(K_MF, N_MF, self.alpha_MF, self.alphaPrior)
                
            ### 4.2 the FM surface
            if len(self.indsFM) > 0:
                X_FM = X[self.indsFM,:]
                # update component indicators
                self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
                # re-order labels!
                self.Z_FM, self.componentsFM = relabel(self.Z_FM, self.componentsFM, self.Kmax)
                # update components
                self.componentsFM = updateGaussianComponents(X_FM, self.Z_FM, self.componentsFM, 
                                                             self.muPrior, self.precisionPrior)
                # update weights
                self.weightFM = updateMixtureWeight(self.Z_FM, self.alpha_FM, self.Kmax)
                # update alpha
                K_FM = len(np.unique(self.Z_FM))
                N_FM = len(self.indsFM)
                self.alpha_FM = updateAlpha(K_FM, N_FM, self.alpha_FM, self.alphaPrior)
            
            ### 4.3 the outsiders
            if len(self.inds0) > 0:
                X_0 = X[self.inds0,:]
                # update component indicators
                self.Z_0 = updateComponentIndicator(X_0, self.weight0, self.components0)
                # re-order labels!
                self.Z_0, self.components0 = relabel(self.Z_0, self.components0, self.Kmax)
                # update components
                self.components0 = updateGaussianComponents(X_0, self.Z_0, self.components0, 
                                                            self.muPrior, self.precisionPrior)
                # update weights
                self.weight0 = updateMixtureWeight(self.Z_0, self.alpha_0, self.Kmax)
                # update alpha
                K_0 = len(np.unique(self.Z_0))
                N_0 = len(self.inds0)
                self.alpha_0 = updateAlpha(K_0, N_0, self.alpha_0, self.alphaPrior)
            
            
            ## 5. Save parameter in chains if...
            if (it >= burn) & ((it+1-burn) % thin == 0):
                self.chains['muL'].append(self.muL)
                self.chains['muD'].append(self.muD)
                self.chains['muNegD'].append(self.muNegD)
                self.chains['gammaL'].append(self.gammaL)
                self.chains['gammaD'].append(self.gammaD)
                self.chains['N_MF'].append(len(self.indsMF))
                self.chains['N_FM'].append(len(self.indsFM))
                self.chains['gamma'].append(self.gamma)
                self.chains['probs'].append(self.probs)
                self.chains['C'].append(self.C)
                self.chains['componentsMF'].append(self.componentsMF)
                self.chains['componentsFM'].append(self.componentsFM)
                self.chains['components0'].append(self.components0)
                self.chains['weightMF'].append(self.weightMF)
                self.chains['weightFM'].append(self.weightFM)
                self.chains['weight0'].append(self.weight0)
                self.chains['alpha_MF'].append(self.alpha_MF)
                self.chains['alpha_FM'].append(self.alpha_FM)
                self.chains['alpha_0'].append(self.alpha_0)
                ## keep track of L and D as well to see what's wrong
                self.chains['L'].append(self.L)
                self.chains['D'].append(self.D)
                
                # 10/26/20: add loglik
                log_lik = self.evalLikelihood()
                self.chains['loglik'].append(log_lik)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}. Log-likelihood={}'.format(it, self.maxIter, log_lik))
            
        return
    
    def plotChains(self, param, s=None, savepath=None):
        
        # updated 10/11/2020: add more plotting functions 
        # (inherited from the DPGMM version)
        
        
        if param.startswith('compo'):
            
            # 10/11/2020: adapted to the 3-surface case
            
            # a helper function for plotting
            def plotSurface(data, name, weights, components, savepath=None):
                # get the min, max range
                Amin = np.min(data); Amax = np.max(data)
                
                # specify range
                Amin = min(15,np.min(data)); Amax = max(50,np.max(data))
            
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
            
                plt.title('predicted log-density of the {} surface'.format(name))
                # 10/26/2020 fix: didn't reverse the order of age pair
                # so plot male age and female age instead
                #plt.xlabel('transmitter age')
                #plt.ylabel('recipient age')
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath)
                plt.show()
                return
            
            # plot each surface one by one
            chain = self.chains[param]
            if s >= len(chain) or s is None:
                s = -1
            components = chain[s]
            
            # the component labels in the relevant iteration
            C = self.chains['C'][s]
            
            # mapping of surface label and name
            #surfs = {0: '0', 1: 'MF', 2: 'FM'}
            
            surfs = {'0': 0, 'MF': 1, 'FM': 2}
            name = param[10:]
            c = surfs[name]
            
            
            # then plot the corresponding component
            data = getPoints(self.E)[C==c,:]
            
            if data.shape[0] == 0:
                pass
            else:
                weights = self.chains['weight'+name][s]
                # make the corresponding plot
                plotSurface(data, name, weights, components, savepath)
            
        elif param=="C":
            # s: can serve as the starting point for querying the chain
            
            # 10/11/2020: adapted to the 3-surface case
            
            def tabulate(C):
                counts = np.empty(shape=3)
                for k in range(3):
                    counts[k] = np.sum(C==k)
                return counts
            
            if s is None or s<0 or s>=len(self.chains['C']):
                s = 0
                
            Cs = np.array(self.chains['C'][s:])
            all_counts = np.apply_along_axis(tabulate, 1, Cs)
            Counts_mean = np.mean(all_counts,axis=0)
            Counts_std = np.std(all_counts,axis=0)
            
            ind = np.arange(len(Counts_mean))
            plt.bar(ind, Counts_mean, 0.5, yerr = Counts_std,
                    error_kw=dict(lw=3, capsize=3, capthick=2))
            plt.title('Number of points allocated to each type through the chain')
            plt.xticks(ind, ('0', 'MF', 'FM'))
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
            
            
        elif param.startswith('prob'):
            chain = np.array(self.chains[param])
            for h in range(3):
                this_label = str(h)
                plt.plot(chain[:,h],"-",label=this_label)
            plt.legend(loc='upper right')
            plt.title('Traceplot of surface allocation probs')
            plt.xlabel('Samples')
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
            
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
    
    def getMeanSurface(self, st=200, en=2000, thin=1, m=15, M=50,
                       plot=True, savepath=None):
        '''
        function to obtain the "mean" surface density for each surface
        '''
        
        # mapping of surface label and name
        surfs = {0: '0', 1: 'MF', 2: 'FM'}
        
        # point grids
        x = np.linspace(m, M)
        y = np.linspace(m, M)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
            
        # 01/09/2021 change: query components with type 
        # (componentsMF, _FM, _0, etc. instead of shared components)
        def calDensity(s, which):
            '''
            get the surface density (in log) at iteration s, for type which(=0,1,2)
            '''
            weights = self.chains['weight'+surfs[which]][s]
            components = self.chains['components'+surfs[which]][s]
            
            Z = evalDensity(XX, weights, components, log=True)
            Z = Z.reshape(X.shape)
            
            return Z
        
        for c in [0,1,2]:
            # go through the selected iterations and accumulate
            for s in range(st,en,thin):
                Z = calDensity(s,c)
                if s == st:
                    dens_c = Z
                else:
                    dens_c = np.concatenate((dens_c, Z), axis=0)
            # change shape       
            dens_c = dens_c.reshape(len(range(st,en,thin)),Z.shape[0],Z.shape[1])
            
            # get mean and variance
            dens_mean = np.apply_along_axis(np.mean, 0, dens_c)
            dens_sd = np.apply_along_axis(np.std, 0, dens_c)
            
            # plot
            if plot:
                ## mean
                plt.contourf(X,Y,dens_mean)
                plt.title('posterior mean log-density of the {} surface'.format(surfs[c]))
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath+surfs[c]+'_mean_density.pdf')
                plt.show()
                
                ## sd
                plt.contourf(X,Y,dens_sd)
                plt.title('posterior log-density std of the {} surface'.format(surfs[c]))
                plt.xlabel('male age')
                plt.ylabel('female age')
                if savepath is not None:
                    plt.savefig(savepath+surfs[c]+'_density_std.pdf')
                plt.show()
                
        return
    
#%%
# update: 10/11/2020
# try running the 3-surface version with DP on the "synthetic real" data

#import pandas as pd
#
#dat = pd.read_csv("../200928_data_not_unlike_real_data.csv")

#%%
# 08/31/2021
# try real Rakai data: 

#import pandas as pd
#dat = pd.read_csv('../Rakai_data.csv')

#%%

# 01/12/2022
# real Rakai data again (with updated info)
import pandas as pd
dat = pd.read_csv('../Rakai_data_Jan2022.csv')

#%%
# 10/26/2020 fix:
# transform scores between interval [0.02, 0.98] to avoid spillover
#def shrinkScore(x, l=0.02, u=0.98):
#    '''
#    x: has to be numpy array
#    l: lower bound
#    u: upper bound
#    '''
#    return x * (u - l) + l

#%%
# 09/12/2021 fix:
# try transforming the MF scores to make them closer to 0 or 1 (except for those in the middle chunk)
#def pushScore(x, l = 0.4, u = 0.6, pw = 2):
#    if x < l:
#        return x**pw
#    elif x > u:
#        return 1-(1-x)**pw
#    else:
#        return x

#%%

# filter out some "low linked prob" data points

# a heuristic fix: only keep those not very close to 0 or 1

#dat = dat[(dat.POSTERIOR_SCORE_LINKED > 0.2) & (dat.POSTERIOR_SCORE_LINKED < 0.98) &
#          (dat.POSTERIOR_SCORE_MF > 0.02) & (dat.POSTERIOR_SCORE_MF < 0.98)]


# 10/26/2020 fix:
# 09/12/2021:
### filter out linkage scores first before shrinking it 
        
# 09/15/2021 update
# re-try real data with the updated treatment of 0/1 scores

# only keep linked scores that are > 0.2
dat = dat[(dat.POSTERIOR_SCORE_LINKED > 0.2)]

## 09/15/2021:
# specially treat points with L=1 or D=1/0
ext_L = np.where(dat.POSTERIOR_SCORE_LINKED == 1)[0]   
ext_D = np.where((dat.POSTERIOR_SCORE_MF == 0) | (dat.POSTERIOR_SCORE_MF == 1))[0]

def_MF = np.where((dat.POSTERIOR_SCORE_LINKED == 1) & (dat.POSTERIOR_SCORE_MF > 0.6))[0]
def_FM = np.where((dat.POSTERIOR_SCORE_LINKED == 1) & (dat.POSTERIOR_SCORE_MF < 0.4))[0]

## 09/20/2021: differentiate between D=0 and D=1 cases when L < 1
## when a pair like this is NOT assigned to 0 surface by the L score, 
## assign it directly to FM if D=0, and to MF if D=1

#L_model_only = np.where(((dat.POSTERIOR_SCORE_MF == 0) | (dat.POSTERIOR_SCORE_MF == 1)) & 
#                        (dat.POSTERIOR_SCORE_LINKED < 1))[0]
potMF = np.where((dat.POSTERIOR_SCORE_MF == 1) & (dat.POSTERIOR_SCORE_LINKED < 1))[0]
potFM = np.where((dat.POSTERIOR_SCORE_MF == 0) & (dat.POSTERIOR_SCORE_LINKED < 1))[0]
D_model_only = np.where((dat.POSTERIOR_SCORE_LINKED == 1) & (dat.POSTERIOR_SCORE_MF < 0.6) & 
                        (dat.POSTERIOR_SCORE_MF > 0.4))[0]

## 09/12/2021:
## also try to remove data with  POSTERIOR_SCORE_MF exactly 0 or 1 
## (there are some people like that making things messy)
## and see if the mess gets resolved

#dat = dat[(dat.POSTERIOR_SCORE_MF < 1) & (dat.POSTERIOR_SCORE_MF > 0)]

# scale the linked scores and male-female scores within (0.02, 0.98)
#dat.POSTERIOR_SCORE_LINKED = shrinkScore(dat.POSTERIOR_SCORE_LINKED)
#dat.POSTERIOR_SCORE_MF = shrinkScore(dat.POSTERIOR_SCORE_MF)

# 09/12/2021:
## try pushing the D scores more towards the 0/1 boundaries to create more separation
# dat.POSTERIOR_SCORE_MF = np.array([pushScore(x, pw=2) for x in dat.POSTERIOR_SCORE_MF])

# leave us with 526 rows in total (filter with L > 0.2 first before shrinking)
print(dat.shape)

L = np.array(dat.POSTERIOR_SCORE_LINKED)
D = np.array(dat.POSTERIOR_SCORE_MF)

print(len(np.where(L > 0.6)[0])) # 365 with L > 0.6
D_sel = D[np.where(L > 0.6)[0]]
print(len(np.where(D_sel > 0.5)[0])) # 198 with D > 0.5 (among those rows with L > 0.6)

## With real data
## after "pre-processing"
## 527 rows, 364 with L > 0.6
## among those L > 0.6, 198 rows with D > 0.5

edges = np.array(dat[['MALE_AGE_AT_MID','FEMALE_AGE_AT_MID']])
nr = edges.shape[0]

E = dict(zip(range(nr), edges))

plt.plot(L,"o")
plt.show()

plt.plot(D, "o") # no super clear pattern
plt.show() 

# plot age pair patterns
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show() 

# save this processed dataset for use in R
#dat.to_csv('../Rakai_data_processed.csv',index=False, index_label=False)
    
#%%

# the prior (shared between fixed and non-fixed allocation inference)

Pr = {"gammaL": {'nu0': 2, 'sigma0': 1}, # the previous default prior setting for score gamma's
      "gammaD": {'nu0': 2, 'sigma0': 1}, # trying this first (to shrink the mixture spread...)
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001,
                'covariance': np.eye(2)*10000},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2), 'Scale': np.eye(2)},
      #"weight": np.ones(K), 
      "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02},
      #"alpha": {'a': 2.0, 'b':3.0}}   
      # 10/26/2020: try a prior for alpha to encourage less shrinkage (large alpha)
      "alpha": {'a': 4.0, 'b': 1.0}} 




#%%


# 01/09/2021: try with fixed muD and muNegD (NOT fixed point allocation)

# 08/31/2021: Do this with real data too (with previous default setting)

#Kmax = 8

# 09/05/2021: UPDATE
## now: have separate priors for gammaL and gammaD
## AND make gammaD prior stronger to try to keep 

## I can make the gammaD prior REALLY strong, but that will squash away the 0 surface completely...
## if it's not ridiculously strong, then pretty much same as before (MF surface vanishes)

model = LatentPoissonDPHGMM(Priors = Pr, K=3, Kmax = 8)

# try a Kmax=2 version
#model = LatentPoissonDPHGMM(Priors = Pr, K=2, Kmax = 2)

# 09/15/2021: try new trick , start with few iters
# 09/20/2021: adjustment with slight argument change in fit function

# 01/12/2022: try re-run with new data (D = +- 1.5)
#             try another one with D =+- 1.0
#             try anohter one with D = +- 1.8

model.fit(E, L, D, samples=3000, burn=0, random_seed = 41, debugHack=False, 
          D_centers = [1.5, -1.5], def_event_inds = [def_MF, def_FM], 
          extreme_inds = [ext_L, ext_D], L_D_model_inds = [potMF, potFM, D_model_only])

model.plotChains('N_MF')
model.plotChains('N_FM')
#model.plotChains('muD')
#model.plotChains('muNegD')
#model.plotChains('weightMF')
#model.plotChains('weightFM')
#model.plotChains('weight0')
model.plotChains('muL')
model.plotChains('gammaD')
model.plotChains('gammaL')
model.plotChains('probs')
#model.plotChains('alpha_MF')

model.plotChains('C', s=1500)

#model.plotChains('componentsMF', s=2500)
#model.plotChains('componentsFM', s=1000)

# plot the components at MAP
## try some diff burn-in options to see what result 
burn = 2000
model.plotChains('componentsMF', s=np.argmax(model.chains['loglik'][burn:])+burn)
model.plotChains('componentsFM', s=np.argmax(model.chains['loglik'][burn:])+burn)
model.plotChains('components0', s=np.argmax(model.chains['loglik'][burn:])+burn)

# plot the "mean" and "std" of log-density on each surface
#model.getMeanSurface(st=500, en=3000, thin=10, m=15, M=50, plot=True, savepath=None)

model.getMeanSurface(st=1000, en=len(model.chains['loglik']), 
                     thin=10, m=15, M=50, plot=True, savepath=None)

# Problems:
# somehow we always have one surface going to zero points....
# sometimes it's 0, sometimes it's FM
# Maybe that's an issue with the separate-3 surface model?


#%%

## 01/09/2021: the version with fixed point allocation
## 08/31/2021: do this on real data with previous default setting
##             also try a Kmax = 10 version

## 09/20/2021: re-run fixed alloc version with this adaptation

## 01/12/2022: re-run fixed alloc version with new data (using previous settings first)

model2 = LatentPoissonDPHGMM(Priors = Pr, K=3, Kmax = 8)

# try a Kmax=2 version
#model = LatentPoissonDPHGMM(Priors = Pr, K=2, Kmax = 2)

model2.fit(E, L, D, samples=3000, burn=0, random_seed = 83, debugHack=False, 
           fixed_alloc= True, def_event_inds = [def_MF, def_FM], 
          extreme_inds = [ext_L, ext_D], L_D_model_inds = [potMF, potFM, D_model_only])

model2.plotChains('N_MF')
model2.plotChains('N_FM')
model2.plotChains('muD')
model2.plotChains('muNegD')
#model.plotChains('weightMF')
#model.plotChains('weightFM')
#model.plotChains('weight0')
model2.plotChains('muL')
model2.plotChains('gammaD')
model2.plotChains('gammaL')
model2.plotChains('probs')
model2.plotChains('alpha_MF')

model2.plotChains('C', s=2500)

#model2.plotChains('componentsMF', s=2500)
# model2.plotChains('componentsFM', s=2500)

# plot the components at MAP
## 08/31/2021: save plots of what I got today...
## 09/20/2021: add burn-in for the MAP estimates
burn = 1000
model2.plotChains('componentsMF', s=np.argmax(model2.chains['loglik'][burn:])+burn) # --> THIS looks reasonable!!!!
model2.plotChains('componentsFM', s=np.argmax(model2.chains['loglik'][burn:])+burn)
model2.plotChains('components0', s=np.argmax(model2.chains['loglik'][burn:])+burn)

# plot the "mean" and "std" of log-density on each surface
model2.getMeanSurface(st=1000, en=len(model2.chains['componentsMF']), thin=10, m=15, M=50, plot=True)#, 
                      #savepath='../Aug31_realData_fixThres_')

#%%
# save results

import pickle as pkl

#pkl.dump(model, file=open("Oct11_synData_3surfaceDP_3000iters.pkl",'wb')) 

pkl.dump(model, file=open("Oct26_synData_3surfaceDP_Kmax2_3000iters.pkl",'wb'))

pkl.dump(model, file=open("Oct26_synData_3surfaceDP_Kmax2_muD0.5_3000iters.pkl",'wb'))

# if fix muD = 0.5, muNegD = -0.5, it works kind of fine (both surface stable)
#model2 = pkl.load(open("Oct26_synData_3surfaceDP_Kmax2_muD0.5_3000iters.pkl",'rb'))

pkl.dump(model, file=open("Oct28_synData_3surfaceDP_Kmax8_muDrestrict_3000iters.pkl",'wb'))

# 01/09/2021: save the fixing threshold version
pkl.dump(model2, file=open('Jan09_synData_fixThres_3surfaceDP_3000iters.pkl', 'wb'))


#%%

# 08/31/2021: save results from [real data] experiments
import pickle as pkl

# with FIXED allocation.....
pkl.dump(model2, file=open('Aug31_realData_fixThres_3000iters.pkl', 'wb'))

## save another try...
pkl.dump(model2, file=open('Aug31_realData_fixThres_3000iters_try2.pkl', 'wb'))

# 09/20/2021: save today's version
pkl.dump(model2, file=open('Sep20_realData_fixThres_3000iters.pkl', 'wb'))

# 01/12/2022: save re-run with fixed alloc
pkl.dump(model2, file=open('Jan12_realData_fixThres_3000iters.pkl', 'wb'))

# with D centers fixed (but not point allocation)
## D centers = +- 0.8
pkl.dump(model, file=open('Aug31_realData_D0.8_2000iters.pkl', 'wb'))

## D centers = +- 1.0
pkl.dump(model, file=open('Aug31_realData_D1.0_2000iters.pkl', 'wb'))

## 09/15/2021 try special treatment of 0/1 scores
pkl.dump(model, file=open('Sep15_realData_specTreat_D1.5_3000iters.pkl', 'wb'))

## 09/15/2021 another try
pkl.dump(model, file=open('Sep15_realData_specTreat_D1.6-1.4_2500iters.pkl', 'wb'))

## 09/21/2021 try again
pkl.dump(model, file=open('Sep20_realData_specTreat_D1.6-1.4_2500iters.pkl', 'wb'))

## another one
pkl.dump(model, file=open('Sep20_realData_specTreat_D1.5_3000iters.pkl', 'wb'))

## 01/12/2022: save D centers = +- 1.5 case
pkl.dump(model, file=open('Jan12_realData_specTreat_D1.5_3000iters.pkl', 'wb'))

## also save D = +- 1.8 case even if it's not that good
pkl.dump(model, file=open('Jan12_realData_specTreat_D1.8_3000iters.pkl', 'wb'))

## 01/15/2022
## same another  D centers = +- 1.5 case (a different run)
pkl.dump(model, file=open('Jan15_realData_specTreat_D1.5_3000iters.pkl', 'wb'))



#%%

# a new function to get surface estimate density function at specified grid points
def getGridDensity(chains, surface, which, lb=15.5, ub=49.5, num = 35, log=False,
                   st = 1000, thin = 10):
    '''
    chains: the chains object of a "model"
    surface: which surface to get (‘0’, ‘MF’, or ‘FM’)
    which: ['map', 'mean', s] where s is the # iteration  (map = Max. Posterior)
    
    log: if on the log scale
    st: the starting index for "mean"
    thin: thinning param for "mean"
    
    return: a matrix of density values evaluated at (x,y)'s
    
    '''
    
    # mapping of surface label and name
    surfs = {0: '0', 1: 'MF', 2: 'FM'}
        
    # point grids
    x = np.linspace(lb, ub, num)
    y = np.linspace(lb, ub, num)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
            
    # density calculation function (adapted from the model methods)
    def calDensity(s):
        '''
        get the surface density (in log) at iteration s, for type which(=0,1,2)
        '''
        weights = chains['weight'+surface][s]
        components = chains['components'+surface][s]
            
        Z = evalDensity(XX, weights, components, log=log)
        Z = Z.reshape(X.shape) # this reshapes Z to the wide matrix shape
            
        return Z
    
    if which == 'map':
        # get MAP estimate
        s = np.argmax(chains['loglik'])
        return calDensity(s)
    elif which == 'mean':
        # get the mean density
        en = len(chains['C'])
        for s in range(st,en,thin):
            Z = calDensity(s)
            if s == st:
                dens = Z
            else:
                dens = np.concatenate((dens, Z), axis=0)
        # change shape       
        dens = dens.reshape(len(range(st,en,thin)),Z.shape[0],Z.shape[1])
        
        # get mean and variance
        dens_mean = np.apply_along_axis(np.mean, 0, dens)
        #dens_sd = np.apply_along_axis(np.std, 0, dens)
        return dens_mean
    else:
        # get a specific iter
        assert isinstance(which, np.integer)
        return calDensity(which)
        


#%%
# get density evaluations at grid points (midpoints of each age band)
        
## 01/15/2022: do this for the updated data analysis
        
# 1. get this for fixed allocation first
        
## (1) MAP 
Z_MF = getGridDensity(model2.chains, 'MF', 'map')
np.savetxt('../MF_surface_midpoints_MAP_fixAlloc_Jan2022.txt', Z_MF)

Z_FM = getGridDensity(model2.chains, 'FM', 'map')
np.savetxt('../FM_surface_midpoints_MAP_fixAlloc_Jan2022.txt', Z_FM)

## (2) Mean surface
Z_MF = getGridDensity(model2.chains, 'MF', 'mean')
np.savetxt('../MF_surface_midpoints_mean_fixAlloc_Jan2022.txt', Z_MF)

Z_FM = getGridDensity(model2.chains, 'FM', 'mean')
np.savetxt('../FM_surface_midpoints_mean_fixAlloc_Jan2022.txt', Z_FM)

## 09/16/2021
# 2. get a version of semi-fixed alloc (spec. treat. D=[1.6,-1.4], also D=[1.5,-1.5])
## (1) get a particular iter result that I like
s = np.argmax(model.chains['loglik'])
Z_MF = getGridDensity(model.chains, 'MF', s)
np.savetxt('../MF_surface_midpoints_MAP_specTreat_Jan2022.txt', Z_MF)

Z_FM = getGridDensity(model.chains, 'FM', s)
np.savetxt('../FM_surface_midpoints_MAP_specTreat_Jan2022.txt', Z_FM)

## (2) get the mean surfaces
Z_MF = getGridDensity(model.chains, 'MF', 'mean', st=1000)
np.savetxt('../MF_surface_midpoints_mean_specTreat_Jan2022.txt', Z_MF)

Z_FM = getGridDensity(model.chains, 'FM', 'mean', st=1000)
np.savetxt('../FM_surface_midpoints_mean_specTreat_Jan2022.txt', Z_FM)


#%%

# 01/12/2022:
# another new function to get aggregated density/frequency of transmitters
# for a particular recipient age group
# Note: need to keep the samples & mean and map as well
# default: male transmitter & female recipient

from scipy.stats.contingency import margins

def getTransFreq(chains, surface, 
                 rec_age = [15, 24],
                 sc_age = [15.5, 49.5, 35],
                 log=False,
                 st = 1000, thin = 10):
    '''
    chains: the chains object of a "model"
    surface: which surface to get (‘0’, ‘MF’, or ‘FM’)
    rec_age: [lb, ub] for recipient gender age bounds
    sr_age: [lb, ub, num] for transmitter gender age bounds and num of eval points
    log: if on the log scale
    st: the starting index for "mean"
    thin: thinning param for "mean"
    
    return: a matrix of density values evaluated at (x,y)'s
    
    '''
    
    # mapping of surface label and name
    surfs = {0: '0', 1: 'MF', 2: 'FM'}
        
    # point grids
    x = np.linspace(sc_age[0], sc_age[1], sc_age[2]) # source age
    y = np.linspace(rec_age[0], rec_age[1], num = 31)   # recipient age; default 31 grid points
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
            
    # density calculation function (adapted from the model methods)
    def calDensity(s):
        '''
        get the surface density (in log or not) at iteration s, for type which(=0,1,2)
        '''
        weights = chains['weight'+surface][s]
        components = chains['components'+surface][s]
            
        Z = evalDensity(XX, weights, components, log=log)
        Z = Z.reshape(X.shape) # this reshapes Z to the wide matrix shape
            
        return Z
    
    # get map estimate first
    smap = np.argmax(chains['loglik'])
    dens_map = calDensity(smap)
    
    # then iterate through
    # get marginal dens for source age as well as the mean dens
    en = len(chains['C'])
    for s in range(st,en,thin):
        Z = calDensity(s)
        # get the marginal dens for source age as well
        _, ymar = margins(Z)
        if s == st:
            dens = Z
            sc_dens = ymar
        else:
            dens = np.concatenate((dens, Z), axis=0)
            sc_dens = np.concatenate((sc_dens, ymar), axis=0) # each row is one sample of the marginal
        
    # change shape  
    dens = dens.reshape(len(range(st,en,thin)),Z.shape[0],Z.shape[1])
    # get mean
    dens_mean = np.apply_along_axis(np.mean, 0, dens)
    #dens_sd = np.apply_along_axis(np.std, 0, dens)
    
    # manipulate sc_dens into a data frame
    ## normalize each row to sum to 1 first
    sc_dens_norm = np.apply_along_axis(lambda x: x/np.sum(x), 0, sc_dens)
    ## vectorize it
    sc_dens_vec = sc_dens_norm.reshape((sc_dens_norm.size,))
    ## create a data frame
    dat_dic = {'freq': sc_dens_vec,
               'monte_carlo_id': [i for i in range(1, sc_dens.shape[0]+1) for j in range(sc_age[2])],
               'sc_age': np.tile(x, sc_dens.shape[0]),
               'direction': surface,
               'rec_age_lb': rec_age[0],
               'rec_age_ub': rec_age[1]}
    sc_dens_dat = pd.DataFrame(dat_dic)

    # return everything
    return sc_dens_dat, dens_mean, dens_map
        

#%%
# try it now
    
# 1. male transmitter age freq. for female recipients between some age bounds

# with non-fixed allocation:
sc_dens_MF, _, _ = getTransFreq(model.chains, 'MF', rec_age = [24,26]) # Try a narrow interval of ages
sc_dens_MF.to_csv('MF_sc_dens_young_women.csv', index=False, index_label=False)

# with fixed allocation
sc_dens_MF2, _, _ = getTransFreq(model2.chains, 'MF', rec_age = [24,26])
sc_dens_MF2.to_csv('MF_sc_dens_young_women_fixedAlloc.csv', index=False, index_label=False)

# 2. female transmitter age as well
sc_dens_FM, _, _ = getTransFreq(model.chains, 'FM', rec_age = [29,31]) # Try a narrow interval of ages
sc_dens_FM.to_csv('FM_sc_dens_middle_men30.csv', index=False, index_label=False)

sc_dens_FM2, _, _ = getTransFreq(model2.chains, 'FM', rec_age = [29,31]) # Try a narrow interval of ages
sc_dens_FM2.to_csv('FM_sc_dens_middle_men30_fixedAlloc.csv', index=False, index_label=False)


#%%
# 01/31/2022
# go through the C point allocations to identify points with a lot of type ambiguity

## get indices that are not def FM or MF points
N = 526
amb_inds = [i for i in range(N) if i not in def_FM and i not in def_MF]

def check_amb_points(chains, inds, st = 1000, en = 3000):
    
    tot_iter = en-st+1
    Cs = chains['C'][st:en]
    Carray = copy(np.array(Cs)[:, inds])
    
    Counts = np.apply_along_axis(lambda v: np.array([np.sum(v==k) for k in range(3)]),0, Carray)/tot_iter
    
    # turns out there are a lot of points that change their types back and forth 
    # so 
    check = np.apply_along_axis(lambda v: np.max(v[1:]) <= 0.7 and v[0] <= 0.1, 
                                0, Counts)
    amb_inds = [inds[i] for i in np.where(check)[0]]
    
    # also return the most frequenstly assigned type
    most_type = np.apply_along_axis(np.argmax, 0, Counts)
    
    return Counts.T, amb_inds, most_type, check


#%%
Counts, amb_i, types, check = check_amb_points(model.chains, amb_inds)

## check entropy of those frequencies
from scipy.stats import entropy
entros = np.apply_along_axis(entropy, 1, Counts)

#thres = 0.8 # check for high entropy ones (more balanced proportions among three categories)
## entropy([0.1,0.2,0.7]) = 0.802
## entropy([0.1,0.3,0.6]) = 0.898
## entropy([0.1,0.1,0.8]) = 0.639
np.sum(entros > 0.9) # 41
np.sum(entros > 0.8) # 77 (53 of them mainly assigned to true events)
np.sum(entros > 0.6) # 189 points



## check out those age pairs
Ages = getPoints(E)
Ages[amb_i,:]
                            
## plot those ambiguous points
iis = amb_i
#iis = [i for i in inds if i in np.where(entros > 0.9)[0]]

within_inds = np.where(check)[0]
within_inds = np.where(entros > 0.8)[0]


to_see = 'most' # or 'fixed'
to_see = 'fixed'

if to_see == 'most':

    ## check for mostly assigned category
    i1 = [amb_inds[i] for i in within_inds if i in np.where(types==1)[0]]
    i2 = [amb_inds[i] for i in within_inds if i in np.where(types==2)[0]]
    i3 = [amb_inds[i] for i in within_inds if i in np.where(types==0)[0]]
    
    tit = 'most freq. type'
else:

    ## check for the fixed type category
    i1 = [amb_inds[i] for i in within_inds if amb_inds[i] in np.where(D>0.5)[0] and amb_inds[i] in np.where(L>0.6)[0]]
    i2 = [amb_inds[i] for i in within_inds if amb_inds[i] in np.where(D<0.5)[0] and amb_inds[i] in np.where(L>0.6)[0]]
    i3 = [amb_inds[i] for i in within_inds if amb_inds[i] in np.where(L<=0.6)[0]]
    
    tit = 'fixed assign. type'
 
if i3:
    plt.plot(Ages[i3,0], Ages[i3,1], "o", color='gray', label='non-event')
if i1:
    plt.plot(Ages[i1,0], Ages[i1,1], "o", label='M->F')
if i2:
    plt.plot(Ages[i2,0], Ages[i2,1], "o", color='orange', label='F->M')

plt.xlim([15, 50])
plt.ylim([15, 50])
plt.ylabel('female age')
plt.xlabel('male age')
plt.legend(loc='upper left', title = tit)
plt.show()



   
                    
                                 
                                 
    
    
    
    
    
