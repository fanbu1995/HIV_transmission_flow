#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:35:05 2020

@author: fan
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
            - "gammaScore": prior dictionary for inverse variance of the score models;
                need "nu0" and "sigma0"
            - "alpha": prior for the DP precision; need "a" and "b"
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
        self.probPrior = Priors["probs"]
        self.alphaPrior = Priors['alpha']
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
        self.alpha = None # DP precision for the shared mixture components
        self.params_to_record = ['muL','muD', 'muNegD', 'gammaL', 'gammaD', 
                                 'N_MF', 'N_FM', 'gamma', 'probs', 'C', 
                                 'components', 'weightMF', 'weightFM', 'weight0',
                                 'alpha']
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
        # 3) DP Gaussian components
        ## some initial estimates of alpha
        self.alpha = rng.gamma(self.alphaPrior['a'], 1/self.alphaPrior['a'], size=1)
        
        X = getPoints(self.E)
        self.components, self.Z = initializeDPGMM(X, self.muPrior, 
                                                  self.precisionPrior, self.K, self.Kmax)
        # 4) GMM weights
        # 4.1) MF surface
        X_MF = X[self.indsMF,:]
        self.weightMF = updateMixtureWeight(self.Z[self.indsMF], self.alpha, self.Kmax)
        
        # 4.2) the FM surface
        X_FM = X[self.indsFM,:]
        self.weightFM = updateMixtureWeight(self.Z[self.indsFM], self.alpha, self.Kmax)
        # 4.2) the outsiders
        self.inds0 = np.where(self.C == 0)[0]
        X_0 = X[self.inds0,:]
        self.weight0 = updateMixtureWeight(self.Z[self.inds0], self.alpha, self.Kmax)
        
        # 5) update alpha
        K = len(np.unique(self.Z))
        self.alpha = updateAlpha(K, N, self.alpha, self.alphaPrior)
        
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
            ### 4.0 the part shared by everyone
            # 10/11/2020: re-order the labels first!
            self.Z, self.components = relabel(self.Z, self.components, self.Kmax)
            self.components = updateGaussianComponents(X, self.Z, self.components, 
                                                       self.muPrior, self.precisionPrior)

            ## UPDATE 10/11/2020: if any set is empty, don't update
            # 4.1 MF surface
            if len(self.indsMF) > 0:
                X_MF = X[self.indsMF,:]
                self.Z[self.indsMF] = updateComponentIndicator(X_MF, self.weightMF, self.components)
                self.weightMF = updateMixtureWeight(self.Z[self.indsMF], self.alpha, self.Kmax)
            # 4.2 the FM surface
            if len(self.indsFM) > 0:
                X_FM = X[self.indsFM,:]
                self.Z[self.indsFM] = updateComponentIndicator(X_FM, self.weightFM, self.components)
                self.weightFM = updateMixtureWeight(self.Z[self.indsFM], self.alpha, self.Kmax)
            # 4.3 the outsiders
            if len(self.inds0) > 0:
                X_0 = X[self.inds0,:]
                self.Z[self.inds0] = updateComponentIndicator(X_0, self.weight0, self.components)
                self.weight0 = updateMixtureWeight(self.Z[self.inds0], self.alpha, self.Kmax)

            # 4.4 update alpha together
            K = len(np.unique(self.Z))
            self.alpha = updateAlpha(K, N, self.alpha, self.alphaPrior)
            
            
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
                self.chains['alpha'].append(self.alpha)
                
                if verbose:
                    print('Parameters saved at iteration {}/{}.'.format(it, self.maxIter))
            
        return
    
    def plotChains(self, param, s=None, savepath=None):
        
        # updated 10/11/2020: add more plotting functions 
        # (inherited from the DPGMM version)
        
        
        if param.startswith('compo'):
            
            # 10/11/2020: adapted to the 3-surface case
            
            # a helper function for plotting
            def plotSurface(dat, name, weights, components, savepath=None):
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
            
                plt.title('predicted log-density of the {} surface'.format(name))
                plt.xlabel('transmitter age')
                plt.ylabel('recipient age')
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
            surfs = {0: '0', 1: 'MF', 2: 'FM'}
            
            for c in range(3):
                # get the relevant data
                if c%2 == 0:
                    data = getPoints(self.E)[C==c,:]
                else:
                    # if FM surface, need to reverse age pair order
                    data = getPoints(self.E)[C==c,:][:,(1,0)]
                    
                name = surfs[c]
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
    
    
#%%
# update: 10/11/2020
# try running the 3-surface version with DP on the "synthetic real" data

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

plt.plot(D, "o") # no super clear pattern
plt.show()     
    
    
#%%

#K = 10

Pr = {"gammaScore": {'nu0': 2, 'sigma0': 1},
      "muGMM": {'mean': np.array([0,0]), 'precision': np.eye(2)*.0001,
                'covariance': np.eye(2)*10000},
      "precisionGMM": {'df': 2, 'invScale': np.eye(2), 'Scale': np.eye(2)},
      #"weight": np.ones(K), 
      "probs": np.ones(3),
      "gammaPP": {'n0': 1, 'b0': 0.02},
      "alpha": {'a': 2.0, 'b':3.0}}   

model = LatentPoissonDPHGMM(Priors = Pr, K=3, Kmax = 10)

model.fit(E, L, D, samples=3000, burn=0, random_seed = 89, debugHack=False)

model.plotChains('N_MF')
model.plotChains('N_FM')
model.plotChains('weightMF')
model.plotChains('weightFM')
model.plotChains('weight0')
model.plotChains('muL')
model.plotChains('muD')
model.plotChains('muNegD')
model.plotChains('gammaD')
model.plotChains('probs')
model.plotChains('alpha')

model.plotChains('C', s=200)

model.plotChains('components', s=2800, savepath='3surfaceDP_components.pdf')

# what's happening:
# DP tends to select very few components: 
#     1 or 2 giant components on each surface (not much pattern)
# again, all points move to one surface (this time MF, not FM - unstable behavior)
# is there an un-identifiability issue??