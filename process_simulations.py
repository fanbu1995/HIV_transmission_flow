#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:54:48 2021

@author: fan
"""

#%%
# process simulation results 
import os
os.chdir('/Users/fan/Documents/Research_and_References/HIV_transmission_flow/')

from copy import copy#, deepcopy

import matplotlib.pyplot as plt

#from utilsHupdate import *

import numpy as np

#import pandas as pd
from copy import copy

# 10/11/2020 update
# numpy.random new generator...
from numpy.random import default_rng
rng = default_rng()

import pickle as pkl


#%%

# plot the posterior means of weightMFs (the weights on the largest two components: 25 vs 35)

def poolWeights(res_dir, nums, last=500, savepath=None):
    weight_means = np.zeros((len(nums), 2))
    for i in range(len(nums)):
        n = nums[i]
        fpath = res_dir + 'weightMF_'+str(n)+'.pkl'
        W = pkl.load(file=open(fpath,'rb'))
        chain = np.array(W[-last:])[:,:2]
        w_means = np.mean(chain, axis=0)
        weight_means[i,:] = w_means
        
    if max(nums) < 200:
        # put weight on younger men to the left side
        weight_means = weight_means[:,::-1]
        
    data = [weight_means[:,0], weight_means[:,1]]
    
    plt.figure(figsize=(6,4))
        
#    bplot = plt.boxplot(data, vert = True, patch_artist=True,
#                labels=['younger','older'])
    
    vplot = plt.violinplot(data, showmeans=True, widths=0.5)
    
    plt.grid(axis='y')
    plt.xticks(np.arange(4), ['','younger', 'older',''])
    plt.title('Proportions of transmissions from younger v.s. older men')
        
    # make the whiskers more obvious
    vplot['cmins'].set_linewidths(3)
    vplot['cmins'].set_color('black')
    vplot['cmaxes'].set_linewidths(3)
    vplot['cmaxes'].set_color('black')
    vplot['cmeans'].set_linewidths(3)
    vplot['cmeans'].set_color('black')
    vplot['cbars'].set_linewidths(3)
    vplot['cbars'].set_color('black')
    
#    colors = ['pink', 'lightblue']
#    for patch, color in zip(bplot['boxes'], colors):
#        patch.set_facecolor(color)
        
    if savepath is not None:
        plt.savefig(savepath)  
    plt.show()
        
    
    return

#%%
    
poolWeights('trans_flow/', list(range(1,101)), savepath='weightsMF_less_youner_men.pdf')

poolWeights('trans_flow/', list(range(101,201)), savepath='weightsMF_more_youner_men.pdf')


#%%

# plot the proportions of transmission events
def poolCs(res_dir, nums, last=500, savepath=None):
    
    def tabulate(C):
        counts = np.empty(shape=2)
        for k in range(1,3):
            counts[k-1] = np.sum(C==k)/np.sum(C!=0)
        return counts
    
    C_means = np.zeros((len(nums), 2))
    
    for i in range(len(nums)):
        n = nums[i]
        fpath = res_dir + 'C_'+str(n)+'.pkl'
        Cs = pkl.load(file=open(fpath,'rb'))[-last:]
        
        all_counts = np.apply_along_axis(tabulate, 1, Cs)
        Counts_mean = np.mean(all_counts,axis=0)
        
        C_means[i,:] = Counts_mean
        
    data = [C_means[:,0],C_means[:,1]]
    plt.figure(figsize=(6,4))
        
#    bplot = plt.boxplot(data, vert = True, patch_artist=True,
#                labels=['younger','older'])
    
    vplot = plt.violinplot(data, showmeans=True, widths=0.5)
    
    plt.grid(axis='y')
    plt.xticks(np.arange(4), ['','M->F', 'F->M',''])
    plt.title('Proportions of identified transmission events')
    
    plt.ylim(0.3,0.7)
    
    # make the whiskers more obvious
    vplot['cmins'].set_linewidths(3)
    vplot['cmins'].set_color('black')
    vplot['cmaxes'].set_linewidths(3)
    vplot['cmaxes'].set_color('black')
    vplot['cmeans'].set_linewidths(3)
    vplot['cmeans'].set_color('black')
    vplot['cbars'].set_linewidths(3)
    vplot['cbars'].set_color('black')
    
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
    return


#%%
poolCs('trans_flow/', list(range(1,101)), savepath='props_trans_events_equal.pdf')

poolCs('trans_flow/', list(range(101,201)), savepath='props_trans_events_moreMF.pdf')

#%%
def plotProps(Cs, savepath=None):
    
    def tabulate(C):
        counts = np.empty(shape=2)
        for k in range(1,3):
            counts[k-1] = np.sum(C==k)/np.sum(C!=0)
        return counts
                
    #Cs = np.array(self.chains['C'][s:])
    all_counts = np.apply_along_axis(tabulate, 1, Cs)
    Counts_mean = np.mean(all_counts,axis=0)
    Counts_std = np.std(all_counts,axis=0)
    Counts_CI = np.quantile(all_counts,[0.025,0.975], axis=0)
    
    ## subtract the mean to get the errorbar width
    Counts_bars = copy(Counts_CI)
    Counts_bars[0,:] = Counts_mean - Counts_bars[0,:]
    Counts_bars[1,:] = Counts_bars[1,:] - Counts_mean
    
    ind = np.arange(len(Counts_mean))
    plt.bar(ind, Counts_mean, 0.4, yerr = list(Counts_bars),
            error_kw=dict(lw=3, capsize=10, capthick=3), 
            color=["#F8766D", "#7CAE00"])
#    plt.errorbar(ind, Counts_mean, yerr = list(Counts_bars), 
#                 fmt='o', capthick=5)
    plt.title('Proportions of transmission events (w/ 95% CI)')
    plt.xticks(ind, ('MF', 'FM'))
    plt.ylim(0,0.65)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
    return Counts_CI


#  (2) plot weights of younger men vs older men
def plotWeights(W, top=2, oldInd=0, savepath=None):
    # default: only plot the top 2 components (one is younger, one is older)
    # W: model.chain['weightMF'], for example
    chain = np.array(W)[:,:top]
    w_means = np.mean(chain, axis=0)
    w_CI = np.quantile(chain, [0.025,0.975], axis=0)
    
    w_bars = copy(w_CI)
    w_bars[0,:] = w_means - w_bars[0,:]
    w_bars[1,:] = w_bars[1,:] - w_means
    
    if oldInd == 0:
        # put "younger" to the left
        w_bars = w_bars[:,::-1]
        w_means = w_means[::-1]
    
    ind = np.arange(len(w_means))
    plt.bar(ind, w_means, 0.4, yerr=list(w_bars),
            error_kw=dict(lw=3, capsize=10, capthick=3),
            color=["#7CAE00","#00BFC4"])
    plt.title('Proportions of transmissions from younger v.s. older men (w/ 95% CI)')
    plt.xticks(ind, ('younger','older'))
         
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()          
    
    return
