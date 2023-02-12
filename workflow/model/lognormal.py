import sys

'''
sys.path.append('../omico')
    
import omico as om
    
from omico import plot as pl
from omico import fit as ft
from omico import analysis as an
from omico import table as tb
'''

import pandas as pd
import numpy as np
import scipy as sp
import random
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fsolve
from scipy.optimize import fsolve

def tln_moments(a,mu,sigma,n):
    
    l = np.exp(n*mu+0.5*(n*sigma)**2) 
    phi = gaussian_cumulative( (n*sigma-a-mu)/sigma )/gaussian_cumulative( (-a-mu)/sigma )
    
    return l*phi

def gaussian_cumulative(z):
    
    return 0.5 * ( 1 + sp.special.erf(z/np.sqrt(2)) )

def gaussian(z):
    
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5*z**2)

def tlnml(D,epsilon=1e-5):
    
    a = np.log(D).min()
    
    def L(p):
        
        mu, sigma = p
        
        m1 = np.mean(np.log(D))
        m2 = np.var(np.log(D))
        
        alpha=(a-mu)/sigma
        phi = gaussian(alpha) / gaussian_cumulative(-alpha)
        
        M = phi*sigma + mu - m1
        S = sigma - np.sqrt(m2 + (mu-m1)*(a-m1))
        
        return (M,S)
    
    return L

def bootstrap_likelihood(R,freqs=None,bins=None,chi=False,s=100,p=0.75):

    r = R.shape[0]

    if chi==True: 
        fields = ['loc','scale','chi']
    else:
        fields = ['loc','scale']
        
    Z=pd.DataFrame(index=range(s),columns=fields)
        
    for i in range(s):
        
        x = np.random.choice(a=R,size=int(p*r))
        
        Z.loc[i,'loc'],Z.loc[i,'scale'] = fsolve(tlnml(x), ( np.mean(np.log(x)), np.var(np.log(x))) )
        
        if chi==True:

            pdf_model = stats.norm.pdf(bins, loc=Z.loc[i,'loc'],scale=Z.loc[i,'scale'])
            Z.loc[i,'chi']=((freqs-pdf_model)**2/pdf_model).mean()

    return Z

"""
def lognormal_mad(X,IMG_DIR,abins=[-3,9],ylim=[-0.15,1.15],legends=False,color={},rare_cut=-np.inf,m_cut=-np.inf,suffix='',n_bins=20):

    ############################################
    ##### DEFAULT 

    if os.path.isfile(IMG_DIR): pass
    else: subprocess.run(['mkdir',IMG_DIR])
        
    K = X.keys()
    C = {k:pl.random_rgb() for k in K}

    for k in K:
        if color!={}: C[k]=color[k]

    ############################################
    ##### PATTERN

    R={}
    B={}
    F={}
    Chi=pd.DataFrame(index=K,columns=['chi-n','chi-bst'])

    #####
    # plot data and the standard lognormal fit
    fig, ax = plt.subplots(ncols=len(K), figsize=(10*len(K),8), gridspec_kw={'wspace':0.2})
    fig.patch.set_facecolor('#FFFFFF')
    
    for i in range(len(K)):
        ax[i].set_facecolor('#FFFFFF')
        ax[i].tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction='out')
        for axis in ['top','bottom','left','right']:  ax[i].spines[axis].set_linewidth(3)
        ax[i].set_facecolor('#FFFFFF')
    
    a_bins = np.linspace(abins[0],abins[1],num=n_bins)
    
    for k,i in zip(K,range(len(K))):
        
        R[k] = X[k]['original mean']['original']
        R[k] = R[k][np.log(R[k])>rare_cut]
        #Z = (np.log(R[k]) - np.log(R[k]).mean())/(np.log(R[k]).std())
        #R[k] = np.exp(Z)
        freqs, bins, ignore = ax[i].hist(np.log(R[k]),bins=a_bins,alpha=0.5,density=True,color=C[k],label='{:0.3e}'.format(k),histtype='stepfilled')
        bins = 0.5*(bins[:-1]+bins[1:])
        ax[i].hist(np.log(R[k]),bins=a_bins,alpha=0.5,density=True,histtype='step',color=C[k],linewidth=3)
        
        B[k]=bootstrap_likelihood(R=R[k].values,freqs=freqs,chi=True,bins=bins,s=500)
        B[k]=B[k][B[k]['mean']>m_cut]
        F[k]=B[k].mean(axis=0)
        
        ab = 0.5*(a_bins[:-1]+a_bins[1:])
        #partition = em.gaussian_cumulative()
        
        normal = stats.norm.pdf(ab, loc=np.log(R[k]).mean(),scale=np.log(R[k]).std())
        Chi.loc[k,'chi-n']=((freqs-normal)**2/normal).mean()
        ax[i].plot(ab,normal,color=C[k],label='std-log-n',linewidth=3,ls='--',alpha=0.4)
        
        normal_bst = stats.norm.pdf(ab, loc=F[k]['mean'],scale=F[k]['sigma'])
        Chi.loc[k,'chi-bst']=((freqs-normal_bst)**2/normal_bst).mean()
        ax[i].plot(ab,normal_bst,color=C[k],label='bootstrap',linewidth=3)
         
        print(np.log(R[k].min()),np.log(R[k].max()))
        
        ax[i].set_xlim(np.log(R[k].min()),np.log(R[k].max()))
        if legends==True: 
            ax[i].legend()
            
        ax[0].set_xlabel('Mean Abundance, $ \log v$',fontsize=25)
        ax[0].set_ylabel('Probability density, $ P( \log v)$',fontsize=25)
            
    fig.savefig(os.path.join(IMG_DIR,'MAD.png'), transparent=True, dpi=150,bbox_inches='tight')
    
    return B,F

def universal_lognormal_mad(X,IMG_DIR,abins=[-3,9],ylim=[-0.15,1.15],legends=False,color={},rare_cut=-np.inf,m_cut=-np.inf,suffix='',n_bins=20):

    ############################################
    ##### DEFAULT 

    if os.path.isfile(IMG_DIR): pass
    else: subprocess.run(['mkdir',IMG_DIR])
        
    K = X.keys()
    C = {k:pl.random_rgb() for k in K}

    for k in K:
        if color!={}: C[k]=color[k]

    ############################################
    ##### PATTERN

    R={}
    B={}
    F={}
    Chi=pd.DataFrame(index=K,columns=['chi','chi-bst'])

    #####
    # plot data and the standard lognormal fit
    fig, ax = plt.subplots(ncols=len(K), figsize=(10*len(K),8), gridspec_kw={'wspace':0.2})
    fig.patch.set_facecolor('#FFFFFF')
    
    for i in range(len(K)):
        ax[i].set_facecolor('#FFFFFF')
        ax[i].tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction='out')
        for axis in ['top','bottom','left','right']:  ax[i].spines[axis].set_linewidth(3)
        ax[i].set_facecolor('#FFFFFF')
    
    a_bins = np.linspace(-4,4,num=n_bins)
    
    for k,i in zip(K,range(len(K))):
        
        R[k] = X[k]['original mean']['original']
        #R[k] = R[k][np.log(R[k])>rare_cut]
        
        Z = (np.log(R[k]) - np.log(R[k]).mean())/(np.log(R[k]).std())
        Z = Z[Z>rare_cut]
        freqs, bins, ignore = ax[i].hist(Z,bins=a_bins,alpha=0.5,density=True,color=C[k],label='{:0.3e}'.format(k),histtype='stepfilled')
        bins = 0.5*(bins[:-1]+bins[1:])
        ax[i].hist(Z,bins=a_bins,alpha=0.9,density=True,histtype='step',color=C[k],linewidth=3)
        
        B[k]=bootstrap_likelihood(R=np.exp(Z),freqs=freqs,chi=True,bins=bins,s=500)
        B[k]=B[k][B[k]['mean']>m_cut]
        F[k]=B[k].mean(axis=0)
        
        ab = 0.5*(a_bins[:-1]+a_bins[1:])
        # occhio nella normal ci va un fattore di funzione di partizione!!!!
        
        zb,za = (Z.max()-F[k]['mean'])/F[k]['sigma'],(Z.min()-F[k]['mean'])/F[k]['sigma']
        partition = gaussian_cumulative(zb)-gaussian_cumulative(za)
        
        normal_bst = stats.norm.pdf(ab, loc=F[k]['mean'],scale=F[k]['sigma'])/partition
        Chi.loc[k,'chi-bst']=((freqs-normal_bst)**2/normal_bst).mean()
        ax[i].plot(ab,normal_bst,color=C[k],label='bootstrap',linewidth=3)
        
        u = np.linspace(-4,4)
        normal = stats.norm.pdf(bins, loc=0,scale=1)
        Chi.loc[k,'chi']=((freqs-normal)**2/normal).mean()
        
        
        ax[i].plot(u,stats.norm.pdf(u, loc=0,scale=1),label='std-log-n',linewidth=3,color='#000000',alpha=0.8)
        ax[i].set_ylim([0,0.7])
        ax[i].set_xlim([-4,4])
        if legends==True: 
            ax[i].legend()
            
    ax[0].set_xlabel('Mean Abundance, $ \log v$',fontsize=25)
    ax[0].set_ylabel('Probability density, $ P( \log v)$',fontsize=25)
        
            
    fig.savefig(os.path.join(IMG_DIR,'universal_MAD.png'), transparent=True, dpi=150,bbox_inches='tight')
    
    return B,F,Chi

def lognormal_params(X,cuts,b,q,norm_chi,engine,SESSION_DIR,normalize=False,suffix=''):
    
    fields = ['mean','sigma','chi']

    W=pd.DataFrame(index=cuts)

    for c in cuts:
        W.loc[c,'skew'] = sp.stats.skew(np.log10(X[c]['original mean']))[0]
        for f in fields:
            W.loc[c,'E'+f]=b[c][f].mean()
            W.loc[c,'D'+f]=b[c][f].std()
            
    if normalize==True:
        W['Echi']=np.log10((norm_chi['chi-bst']/norm_chi['chi']).astype(float))#((W['Echi']/norm_chi).astype(float))
        W['Dchi']=np.zeros(len(W.index))#((W['Echi']/norm_chi**2).astype(float))

    fig, ax = plt.subplots(ncols=4, figsize=(48,10), gridspec_kw={'wspace':0.2})
    fig.patch.set_facecolor('#FFFFFF')
    
    ax[3].plot(cuts,W['skew'],color=pl.random_rgb(),linewidth=3)

    for i,f in zip(range(3),fields):
        ax[i].set_facecolor('#FFFFFF')
        ax[i].tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction='in')
        for axis in ['top','bottom','left','right']:  ax[i].spines[axis].set_linewidth(3)
        ax[i].set_facecolor('#FFFFFF')

        ax[i].errorbar(x=cuts,y=W['E'+f],yerr=W['D'+f],color=pl.random_rgb(),linewidth=10)
        ax[i].set_ylabel(f,fontsize=35)
        
        if engine!='Core-Kaiju': 
            print(engine)
            ax[i].set_xscale('log')
            ax[i].set_xlim(1e-8,1e-2)

    ax[3].tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction='in')
    for axis in ['top','bottom','left','right']:  ax[3].spines[axis].set_linewidth(3)
        
    ax[3].plot(cuts,W['skew'],color=pl.random_rgb(),linewidth=10)
    ax[3].set_ylabel('skewness',fontsize=35)
    
    bound = np.linspace(cuts.min(),cuts.max())
    #ax[3].plot(bound,np.zeros(bound.shape[0]),ls='--',linewidth=3,color='#000000')
    
    #ax[2].set_yscale('log')
    ax[2].set_ylabel('$\log_{10} \\chi_{Trun} / \\chi_{Norm}$')

    if engine=='Core-Kaiju':
        ax[1].set_xlabel('Core-PFAM cut-off, $\\phi$',fontsize=35)
    else:
        ax[1].set_xlabel('RA cut-off, $\\kappa$',fontsize=35)    
        ax[3].set_xscale('log')
        ax[3].set_xlim(1e-8,1e-2)

    fig.savefig(os.path.join(SESSION_DIR,f'log-normal-params_{suffix}.png'), transparent=True, dpi=150,bbox_inches='tight') 
    
    return W
"""