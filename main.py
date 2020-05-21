#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:23 2020

@author: fan
"""

#%%
import os
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

#%%

# the original model and inference method

from utils import *

class LatentPoissonGMM:
    def __init__(self, Priors, K=3, linkThreshold=0.6):
        '''
        Inialize an instance of the LatentPoissonGMM model;
        Priors: a big dictionary of all the priors
            - "gammaPP": prior dictionary for the NHPP scale (gamma); 
                need "n0" and "b0"
            - "muGMM": prior dictionary for the means in Gaussian Mixture; 
                need "mean" and "precision"
            - "precisionGMM": prior dictionary for the precision matrices in Gaussian Mixture;
                need "df" and "invScale"
            - "weight": prior vector (length K) for Gaussian Mixture weight;
            - "gammaScore": prior dictionary for inverse variance of the score models;
                need "nu0" and "sigma0"
            
        '''
        self.name = "Latent Poisson Process with Gaussian Mixture density"
        self.linkInitialThreshold = linkThreshold
        # number of mixture components
        self.K = K
        # prior part
        self.ScoreGammaPrior = Priors["gammaScore"]
        self.muPrior = Priors["muGMM"]
        self.precisionPrior = Priors["precisionGMM"]
        self.weightPrior = Priors["weight"]
        self.PPGammaPrior = Priors["gammaPP"]
        # data part
        self.E = None # all the (a_M,a_F) pairs
        self.L = None # all the linked scores
        self.D = None # all the direction scores
        self.indsMF = None # indices on the MF surface
        self.indsFM = None # indices on the FM surface
        self.E_MF = None # event set on MF surface
        self.E_FM = None # event set on FM surface
        # parameters
        self.muL = None
        self.muD = None
        self.muNegD = None
        self.gammaL = None
        self.gammaD = None
        self.gammaMF = None
        self.gammaFM = None
        self.componentsMF = None
        self.componentsFM = None
        self.weightMF = None
        self.weightFM = None
        self.Z_MF = None # component indicator for MF process
        self.Z_FM = None # component indicator for MF process
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gammaMF', 'gammaFM', 
                                 'componentsMF', 'weightMF',
                                 'componentsFM', 'weightFM']
        # log-likelihood
        #self.log-lik-terms = None # each pair's contribution to the log-likelihood
        self.log_lik = None # total log-likelihood
        # posterior inference (summary statistics and chains)
        self.maxIter = None
        self.burn = 0
        self.thin = 1
        self.batch_size = None
        self.accept = dict() # counter for acceptance times
        self.chains = {param: list() for param in self.params_to_record}
            # a dictionary for parameter samples
            
    def evalLikelihood(self, E_MF, E_FM, X_MF=None, X_FM=None, subset=None):
        '''
        Evaluate likelihood
        #E_MF, E_FM: dictionary of points in MF and FM
        X_MF, X_FM: optional; (n,p) arrays of points in MF and FM
        subset: SORTED indices of subset to evaluate on
        Returns total log likelihood (or log-likehood part on the subset)
        '''
        
        if (X_FM is None) or (X_MF is None):
            X_FM = getPoints(E_FM, subset=subset)
            X_MF = getPoints(E_MF, subset=subset)
            
        #print(X_FM)
        #print(X_MF)
        
        # Right now: STUPID WAY - sum over individual entries
        # later might change
        
        LLik = np.sum(evalLLikelihood(self.L, E_MF, E_FM, self.muL, 
                                      self.gammaL, subset=subset, log=True))
        DLik = np.sum(evalDLikelihood(self.D, E_MF, E_FM, self.muD, self.muNegD, 
                                      self.gammaD, subset=subset, log=True))
        
        MFLik = np.sum(evalDensity(X_MF, self.weightMF, self.componentsMF, log=True)) if X_MF.size > 0 else 0
        FMLik = np.sum(evalDensity(X_FM, self.weightFM, self.componentsFM, log=True)) if X_FM.size > 0 else 0
        
        #N_MF = len(E_MF); N_FM = len(E_FM)
        N_MF = X_MF.shape[0]; N_FM = X_FM.shape[0]
        
        ## This is WRONG! NEED TO FIX THIS
        
        total = LLik + DLik + MFLik + FMLik
        
        if subset is None:
            to_add = (N_MF * np.log(self.gammaMF) + N_FM * np.log(self.gammaFM) - 
                      np.log(range(1,N_MF+1)).sum() - np.log(range(1,N_FM+1)).sum())
            total += to_add - (self.gammaMF + self.gammaFM)
            self.log_lik = total
            
        return total
        

    
    def fit(self, E, L, D, samples = 1000, burn = 0, thin = 1, batch_size = 1,
            random_seed = 42, verbose = True, debugHack = False):
        '''
        Fit the model via MCMC
        '''
        # set up
        self.E = E
        self.L = L
        self.D = D
        #self.log-lik-terms = np.empty(len(E))
        self.burn = burn
        self.thin = thin
        self.maxIter = samples * thin + burn
        self.batch_size = batch_size
        
        np.random.seed(random_seed)
        
        # initialize
        # 1) scores
        self.L, inds, self.muL, self.gammaL = initializeLinkedScore(self.L, self.linkInitialThreshold)
        self.D, self.indsMF, self.indsFM, self.muD, self.muNegD, self.gammaD = initializeDirectScore(self.D, inds)
        # 2) the PP
        self.E_MF, self.E_FM, self.gammaMF, self.gammaFM = initializePP(self.E, self.indsMF, self.indsFM)
        # 3) the MF surface
        X_MF = getPoints(self.E_MF)
        self.componentsMF, self.Z_MF = initializeGMM(X_MF, self.K)
        self.weightMF = updateMixtureWeight(self.Z_MF, self.weightPrior)
        # 4) the FM surface
        X_FM = getPoints(self.E_FM)
        self.componentsFM, self.Z_FM = initializeGMM(X_FM, self.K)
        self.weightFM = updateMixtureWeight(self.Z_FM, self.weightPrior)
        
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
                self.muL, self.gammaL = updateLModel(self.L, self.E_MF, self.E_FM, self.muL, 
                                                     self.gammaL, self.ScoreGammaPrior)
                
                self.muD, self.muNegD, self.gammaD = updateDModel(self.D, self.E_MF, self.E_FM, 
                                                                  self.muD, self.muNegD, 
                                                                  self.gammaD, self.ScoreGammaPrior)                
                
            
            
            ## 2. the MF and FM point configurations
            ### make a copy of the current version first
            E_MF_old = copy(self.E_MF); E_FM_old = copy(self.E_FM)
            
            ### then propose change
            self.E_MF, self.E_FM, chosen, step = proposePP(self.E, self.E_MF, self.E_FM, self.batch_size)
            
            ### accept or reject
            ACC = "accepted"
            
#            logLik_old = self.evalLikelihood(E_MF_old, E_FM_old, subset=chosen)
#            logLik_propose = self.evalLikelihood(self.E_MF, self.E_FM, subset=chosen)
            
#            N_MF_old = len(E_MF_old); N_FM_old = len(E_FM_old)
#            N_MF = len(self.E_MF); N_FM = len(self.E_FM)
#            
#            logDiff = (logLik_propose - logLik_old + 
#                       (N_MF - N_MF_old) * np.log(self.gammaMF) + 
#                       (N_FM - N_FM_old) * np.log(self.gammaFM) -
#                       np.sign(N_MF - N_MF_old) * np.log(range(min(N_MF,N_MF_old)+1,max(N_MF,N_MF_old)+1)).sum() -
#                       np.sign(N_FM - N_FM_old) * np.log(range(min(N_FM,N_FM_old)+1,max(N_FM,N_FM_old)+1)).sum())
#            
            # HACK 2: calculate log-lik on the whole dataset instead of on subset...
            logLik_old = self.evalLikelihood(E_MF_old, E_FM_old, subset=None)
            logLik_propose = self.evalLikelihood(self.E_MF, self.E_FM, subset=None)
            logDiff = logLik_propose - logLik_old
            
            # HACK 3: add a "random noise point" density component to the likelihood
            # assume an outside point comes from Unif([15,45] x [15,45])
            # NOTE: this only works with batch_size == 1 case!!
            if step == 'b':
                logDiff -= np.log(1/30 * 1/30)
            elif step == 'd':
                logDiff += np.log(1/30 * 1/30)
            else:
                pass
            
            if logDiff > 0:
                #self.accept += 1
                prob = 1
            else:
                draw = np.random.random_sample()
                prob = np.exp(logDiff)
                if draw < prob:
                    #self.accept += 1
                    pass
                else:
                    self.E_MF = E_MF_old; self.E_FM = E_FM_old
                    ACC = "rejected"
                    
            if ACC == "accepted":
                if step in self.accept:
                    self.accept[step] += 1
                else:
                    self.accept[step] = 1
                    
            if verbose:
                print("In iteration {}, acceptance prob. ={}, proposal type {} gets {}!".format(it, prob, step, ACC))
                
            self.indsMF = list(self.E_MF.keys())
            self.indsFM = list(self.E_FM.keys())
                    
            ## 3. Update gammaMF and gammaFM
            # Hack it for debugging...
            if debugHack:
                self.gammaMF, self.gammaFM = Settings['N_MF'], Settings['N_FM']
            else:
                self.gammaMF, self.gammaFM = updateGammaPP(self.E_MF, self.E_FM, self.PPGammaPrior)
            
            
            ## 4. Update the Gaussian Mixture Model for the densities
            ### MF surface
            X_MF = getPoints(self.E_MF)
            
            # Hack it for debugging...
            if debugHack:
                self.componentsMF = Settings['componentsMF']
                self.weightMF = Settings['weightMF']
                self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
            else:
                self.Z_MF = updateComponentIndicator(X_MF, self.weightMF, self.componentsMF)
                self.componentsMF = updateGaussianComponents(X_MF, self.Z_MF, 
                                                             self.componentsMF, 
                                                             self.muPrior, self.precisionPrior)
                self.weightMF = updateMixtureWeight(self.Z_MF, self.weightPrior)
            
            
            ### FM surface
            X_FM = getPoints(self.E_FM)
            
            # Hack it for debugging...
            if debugHack:
                self.componentsFM = Settings['componentsFM']
                self.weightFM = Settings['weightFM']
                self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
            else:
                self.Z_FM = updateComponentIndicator(X_FM, self.weightFM, self.componentsFM)
                self.componentsFM = updateGaussianComponents(X_FM, self.Z_FM, 
                                                             self.componentsFM, 
                                                             self.muPrior, self.precisionPrior)
                self.weightFM = updateMixtureWeight(self.Z_FM, self.weightPrior)
            
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
                self.chains['componentsMF'].append(self.componentsMF)
                self.chains['componentsFM'].append(self.componentsFM)
                self.chains['weightMF'].append(self.weightMF)
                self.chains['weightFM'].append(self.weightFM)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}.'.format(it, self.maxIter))
            
        return
    
    def plotChains(self, param):
        if param.startswith('compo'):
            # don't deal with components right now...
            pass
        elif param.startswith('weight'):
            chain = np.array(self.chains[param])
            for k in range(self.K):
                plt.plot(chain[:,k],"-",label=str(k))
            plt.legend('upper right')
            plt.show()
        else:
            plt.plot(self.chains[param])
            plt.show()
            
        return
            

#%%

# try running the original model        

Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
      "weight": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02}}  

model = LatentPoissonGMM(Priors = Pr, K=3)

## Some completely made-up data that won't follow the model at all
#E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
#L = (1-0.3)* np.random.random_sample(100) + 0.3
#D = np.random.random_sample(100)
#
#model.fit(E,L,D, samples=2000, burn=0, random_seed = 71)
# for this one: one of the event sets will eventually get empty...

Settings = {'N_MF': 100, 'N_FM': 100, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.4, 0.3, 0.3]), 'weightFM': np.array([0.4, 0.3, 0.3]),
            'componentsMF': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([40,25], np.diag([1/4,1/9]))],
            'componentsFM': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([25,40], np.diag([1/9,1/4]))]}


E, L, D = simulateLatentPoissonGMM(250, Settings)

E_MF = {i:a for i,a in E.items() if i in range(50)}
E_FM = {i:a[::-1] for i,a in E.items() if i in range(50,100)}

# visualize a bit
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show()

plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

# try to fit 
model.fit(E, L, D, samples=3000, burn=0, random_seed = 71, batch_size=1, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF')
model.plotChains('N_FM')
model.plotChains('weightMF')
model.plotChains('weightFM')
model.plotChains('muL')
model.plotChains('muD')

# check proposals
print(model.accept) #{'s': 146, 'd': 206, 'b': 104}

#%%
# check GMM log density
#evalDensity(getPoints(model.E_MF), weight = model.weightMF, components=model.componentsMF)
#
#evalDensity(getPoints(model.E_FM), weight = model.weightFM, components=model.componentsFM)

# all very negative...

#%%
# Test that evalLLikelihood works
#E_MF = {i:a for i,a in E.items() if i in range(50)}
#E_FM = {i:a[::-1] for i,a in E.items() if i in range(50,100)}
#
#all_LL = evalLLikelihood(model.L, E_MF, E_FM, model.muL, model.gammaL)
#
#LL_in = norm(loc=model.muL, scale=1/np.sqrt(model.gammaL)).logpdf(model.L[:100])
#LL_out = norm(loc=0, scale=1/np.sqrt(model.gammaL)).logpdf(model.L[100:])
#
#np.all(all_LL[:100] == LL_in)
#np.all(all_LL[100:] == LL_out)
#
## also try subset
#sel = [1,5,25,81,90,103]
#sel_LL = evalLLikelihood(model.L, E_MF, E_FM, model.muL, model.gammaL, subset=sel)
# seems to work just fine

#%%
## Test that evalDLikelihood works
#all_DL = evalDLikelihood(model.D, E_MF, E_FM, model.muD, model.muNegD, model.gammaD, 
#                         subset=None, log=True)
#LL_MF = norm(loc=model.muD, scale=1/np.sqrt(model.gammaD)).logpdf(model.D[:50])
#LL_FM = norm(loc=model.muNegD, scale=1/np.sqrt(model.gammaD)).logpdf(model.D[50:100])
#LL_out = norm(loc=0, scale=1/np.sqrt(model.gammaD)).logpdf(model.D[100:])
#
#np.all(all_DL == np.concatenate((LL_MF,LL_FM,LL_out)))

#all_DL[50:100] == LL_FM # False!


#%%

# UPDATED model class and inference method       
        
from utilsH import *

class LatentPoissonHGMM:
    def __init__(self, Priors, K=3, linkThreshold=0.6):
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
            - "gammaScore": prior dictionary for inverse variance of the score models;
                need "nu0" and "sigma0"
            
        '''
        self.name = "Latent Poisson Process with Gaussian Mixture density"
        self.linkInitialThreshold = linkThreshold
        # number of mixture components
        self.K = K
        # prior part
        self.ScoreGammaPrior = Priors["gammaScore"]
        self.muPrior = Priors["muGMM"]
        self.precisionPrior = Priors["precisionGMM"]
        self.weightPrior = Priors["weight"]
        self.PPGammaPrior = Priors["gammaPP"]
        self.probPrior = Priors["probs"]
        # data part
        self.E = None # all the (a_M,a_F) pairs
        self.L = None # all the linked scores
        self.D = None # all the direction scores
        self.indsMF = None # indices on the MF surface
        self.indsFM = None # indices on the FM surface
        self.inds0 = None # indices for the outsider points
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
        self.components = None # a joint set of GMM components shared by all 3 surfaces
        self.weightMF = None
        self.weightFM = None
        self.weight0 = None # GMM weights for the "outside" surface
        self.Z = None # component indicator for all points (length N)
#        self.Z_FM = None # component indicator for MF process
#        self.Z_0 = None # component indicator for the outside process
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gamma', 'probs', 'C', 
                                 'components', 'weightMF', 'weightFM', 'weight0']
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
        Returns a length-N vector of indicators (values in 0, 1, 2)
        '''

        N = len(self.E)
        indsall = list(range(N))
        
        condProbs = np.empty((N,3))
        
        # h=0 (all outside)
        condProbs[:,0] = (evalLLikelihood(self.L, [], [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], [], self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weight0, self.components) +
                 np.log(self.probs[0]))
        
        # h=1 (all in MF)
        condProbs[:,1] = (evalLLikelihood(self.L, indsall, [], self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, indsall, [], self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weightMF, self.components) +
                 np.log(self.probs[1]))
        
        # h=2 (all in FM)
        condProbs[:,2] = (evalLLikelihood(self.L, [], indsall, self.muL, self.gammaL) + 
                 evalDLikelihood(self.D, [], indsall, self.muD, self.muNegD, self.gammaD) + 
                 evalDensity(getPoints(self.E), self.weightFM, self.components) +
                 np.log(self.probs[2]))
        
        self.C = np.apply_along_axis(lambda v: choice(range(3), replace=False, 
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
        # 2) the PP
        self.gamma, self.probs = initializePP(self.E, self.indsMF, self.indsFM)
        self.C = np.zeros(N)
        self.C[self.indsMF] = 1
        self.C[self.indsFM] = 2
        # 3) Gaussian components
        X = getPoints(self.E)
        self.components, self.Z = initializeGMM(X, self.K)
        # 4) GMM weights
        # 4.1) MF surface
        X_MF = X[self.indsMF,:]
        self.weightMF = updateMixtureWeight(self.Z[self.indsMF], self.weightPrior)
        # 4.2) the FM surface
        X_FM = X[self.indsFM,:]
        self.weightFM = updateMixtureWeight(self.Z[self.indsFM], self.weightPrior)
        # 4.2) the outsiders
        self.inds0 = np.where(self.C == 0)[0]
        X_0 = X[self.inds0,:]
        self.weight0 = updateMixtureWeight(self.Z[self.inds0], self.weightPrior)
        
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
            
            ## 2.2 update probs
            self.probs = updateProbs(self.C, self.probPrior)
            
            ## 2.3 bookkeeping
            self.indsMF = np.where(self.C == 1)[0]
            self.indsFM = np.where(self.C == 2)[0]
            self.inds0 = np.where(self.C == 0)[0]
            
            #self.E_MF = {pair: age for pair, age in self.E.items() if pair in self.indsMF}
            #self.E_FM = {pair: age for pair, age in self.E.items() if pair in self.indsFM}
            #self.E_0 = {pair: age for pair, age in self.E.items() if pair in inds0}
            
                    
            ## 3. Update gamma
            self.gamma = np.random.gamma(self.PPGammaPrior['n0']+N, 1/(self.PPGammaPrior['b0']+1))
            
            ## 4. Update the Gaussian Mixture Model for the densities
            ### the part shared by everyone
            self.components = updateGaussianComponents(X, self.Z, self.components, 
                                                       self.muPrior, self.precisionPrior)
            
            
            # 4.1 MF surface
            X_MF = X[self.indsMF,:]
            self.Z[self.indsMF] = updateComponentIndicator(X_MF, self.weightMF, self.components)
            self.weightMF = updateMixtureWeight(self.Z[self.indsMF], self.weightPrior)
            # 4.2 the FM surface
            X_FM = X[self.indsFM,:]
            self.Z[self.indsFM] = updateComponentIndicator(X_FM, self.weightFM, self.components)
            self.weightFM = updateMixtureWeight(self.Z[self.indsFM], self.weightPrior)
            # 4.3 the outsiders
            X_0 = X[self.inds0,:]
            self.Z[self.inds0] = updateComponentIndicator(X_0, self.weight0, self.components)
            self.weight0 = updateMixtureWeight(self.Z[self.inds0], self.weightPrior)

            
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
                self.chains['components'].append(self.components)
                self.chains['weightMF'].append(self.weightMF)
                self.chains['weightFM'].append(self.weightFM)
                self.chains['weight0'].append(self.weight0)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}.'.format(it, self.maxIter))
            
        return
    
    def plotChains(self, param):
        if param.startswith('compo'):
            # don't deal with components right now...
            pass
        elif param=="C":
            # don't deal with C indicators either...
            pass
        elif param.startswith('weight'):
            chain = np.array(self.chains[param])
            for k in range(self.K):
                plt.plot(chain[:,k],"-",label=str(k))
            plt.legend('upper right')
            plt.show()
        elif param.startswith('prob'):
            chain = np.array(self.chains[param])
            for h in range(3):
                plt.plot(chain[:,h],"-",label=str(h))
            plt.legend('upper right')
            plt.show()
        else:
            plt.plot(self.chains[param])
            plt.show()
            
        return



 
#%%
# try running the updated new model
 
Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
      "weight": np.ones(3), "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02}}  

model = LatentPoissonHGMM(Priors = Pr, K=3)

## Some completely made-up data that won't follow the model at all
#E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
#L = (1-0.3)* np.random.random_sample(100) + 0.3
#D = np.random.random_sample(100)
#
#model.fit(E,L,D, samples=2000, burn=0, random_seed = 71)
# for this one: one of the event sets will eventually get empty...

Settings = {'N_MF': 100, 'N_FM': 100, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.4, 0.5, 0.1]), 'weightFM': np.array([0.4, 0.1, 0.5]),
            'weight0': np.array([0.1,0.1,0.8]),
            'components': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([40,25], np.diag([1/4,1/9]))]}


E, L, D = simulateLatentPoissonHGMM(250, Settings)

E_MF = {i:a for i,a in E.items() if i in range(100)}
E_FM = {i:a for i,a in E.items() if i in range(100,200)}

# visualize a bit
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show()

plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

# try to fit 
model.fit(E, L, D, samples=3000, burn=0, random_seed = 71, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF')
model.plotChains('N_FM')
model.plotChains('weightMF')
model.plotChains('weightFM')
model.plotChains('weight0')
model.plotChains('muL')
model.plotChains('muD')


#%%
# try running the updated new model
# a harder version
 
Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2)},
      "weight": np.ones(6), "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02}}  

model = LatentPoissonHGMM(Priors = Pr, K=6)

## Some completely made-up data that won't follow the model at all
#E = {i: (np.random.random_sample(),np.random.random_sample()) for i in range(100)}
#L = (1-0.3)* np.random.random_sample(100) + 0.3
#D = np.random.random_sample(100)
#
#model.fit(E,L,D, samples=2000, burn=0, random_seed = 71)
# for this one: one of the event sets will eventually get empty...

Settings = {'N_MF': 100, 'N_FM': 100, 
            'muL': 2, 'muD': 1.5, 'muNegD': -1.5, 
            'gammaL': 1, 'gammaD': 1, 
            'weightMF': np.array([0.6, 0.4, 0, 0, 0, 0]), 
            'weightFM': np.array([0.1, 0.4, 0.5, 0, 0, 0]),
            'weight0': np.array([0, 0, 0, 0.3, 0.3, 0.4]),
            'components': [([40,40], np.diag([1/4,1/4])), ([25,25], np.diag([1/9,1/9])), 
                             ([40,25], np.diag([1/4,1/9])), ([30,30], np.diag([1/9,1/9])),
                             ([35,20], np.diag([1/16,1/16])), ([35,35], np.diag([1/9,1/4]))]}


E, L, D = simulateLatentPoissonHGMM(250, Settings)

E_MF = {i:a for i,a in E.items() if i in range(100)}
E_FM = {i:a for i,a in E.items() if i in range(100,200)}

# visualize a bit
X = getPoints(E)
plt.plot(X[:,0], X[:,1], "o")
plt.show()

plt.plot(X[range(100),0], X[range(100),1], "o")
plt.show()

plt.plot(X[range(100,200),0], X[range(100,200),1], "o")
plt.show()


plt.plot(L,"o")
plt.show()

plt.plot(D, "o")
plt.show()

# try to fit 
model.fit(E, L, D, samples=3000, burn=0, random_seed = 71, debugHack=False)

# plot number of points in each process
model.plotChains('N_MF')
model.plotChains('N_FM')
model.plotChains('weightMF')
model.plotChains('weightFM')
model.plotChains('weight0')
model.plotChains('muL')
model.plotChains('muD')
model.plotChains('probs')