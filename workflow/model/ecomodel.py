from re import T
import sys

import lognormal as ln
import laplace as lp

import pandas as pd
import numpy as np
import random
import subprocess
from scipy import stats

from sklearn import linear_model

import matplotlib.pyplot as plt


sys.path.append('../omico')

import plot as pl
import fit as ft
import analysis as an
import table as tb

def FitTaylorAmplitude(x,y,steps=20,tau=2):
    
    T=pd.DataFrame(columns=['x','y'])
    T['x'],T['y']=x,y

    ux=np.log(x)
    
    Z=np.linspace(ux.min(),.5*(ux.min()-ux.max()),steps)
    
    F=pd.DataFrame(index=Z,columns=['R','A'])
        
    for z in Z:
        
        U=T[np.log(T['x'])>z].copy()
        A=np.log(U['y']/U['x']**tau).mean()
        F.loc[z,'A']=np.exp(A)
        
        U=np.log(U)
        
        F.loc[z,'R']=np.mean((U['y']-tau*U['x']-A)**2)
    
    zw = F.index[np.argmin(F['R'])]
    
    return F.loc[zw,'A']

class CompoundSchlomilch:
    
    def __init__(self,D,transforms=['relative','binary']):
        
        self.data = D
        
        self.std_transforms=transforms
        
        self.data.built_in_transform(which=transforms)
        self.observables = self.data.get_observables(zipf=True,out=True)
        self.observables = self.observables.sort_values(('zipf rank','original'))

    def sample_parameters(self,samples=None,mode='random',scale='original',rank_conservation=False,report=True,verbose=False):
        
        if verbose==False: end=''
        else: end=None

        print(verbose*f"Sampling {self.mad['model']} MAD",end=end)
        
        if self.mad['model']=='log-normal':
            z = np.random.normal(**self.mad['params'],size=samples)

        elif self.mad['model']=='log-laplace':
            z = np.random.laplace(**self.mad['params'],size=samples)

        m = np.exp(z)
        m = m[np.log(m)>self.mad['min_ln']]
        m = m[np.log(m)<self.mad['max_ln']]

        self.mu = np.random.choice(a=m,size=len(self.data.components))     
        
    def sample_model(self,tau=2,ra_cut=-np.inf,rank_conservation=False,intermediate=False):
        
        N = self.data.realization_size
        samples = self.data.samples
        nSamples = len(list(samples))
        gamma = len(list(self.data.components))

        # choose the scaling family
        if self.taylor['exponent']==2:
            self.alpha = np.ones(gamma)/self.taylor['amplitude']
            self.beta  = self.alpha/self.mu
        elif self.taylor['exponent']==1:
            self.beta  = np.ones(gamma)/self.taylor['amplitude']
            self.alpha = self.mu*self.beta
            
        D=np.zeros(self.data.shape)

        for i in range(gamma): 
            D[i,:] = np.random.gamma(shape=self.alpha[i],scale=self.beta[i]**(-1),size=nSamples)
        
        D=D/D.sum(axis=0)

        R = pd.DataFrame(index=self.data.components,columns=self.data.samples,data=np.zeros(self.data.shape))
        R.index.name = self.data.annotation

        for j,s in zip(range(nSamples),samples): 
            R[s] = np.random.multinomial(n=int(N[s]),pvals=D[:,j])
        
        ranked_relative_abundance =(R/R.sum(axis=0)).mean(axis=1).sort_values(ascending=False)
        R=R.loc[ ranked_relative_abundance.index ]

        R.index = self.observables.index
        
        self.sample = tb.table(R,ra_cut=ra_cut)
        
        self.sample.built_in_transform(which=self.std_transforms)
        self.sample_observables = self.sample.get_observables(zipf=True,out=True)
        self.sample_observables = self.sample_observables.sort_values(('zipf rank','original'))
                    
    def fit_mad(self,write=False,model='',scale='relative',mad=None,fit=True,cut_field='mean',min_logma=-np.inf,cut=-50,ensemble=1000,report=False):
        
        if write==False:
            r = self.observables[f'{scale} mean']['original']
            r = r[np.log(r)>min_logma]

            mad = {}
            mad['model']=model

            print(f"Fitting MAD with {model} model")

            if model=='log-normal': 
                M = ln.bootstrap_likelihood(R=r,s=ensemble)
            elif model=='log-laplace': 
                M = lp.bootstrap_likelihood(R=np.log(r),s=ensemble)

            M=M[M[cut_field]>cut]

            mad['params']=M.mean(axis=0).to_dict()

            # three lines to make it more theory like
            # distinguish the dirichlet mode from the schlomilch one!
            mad['min_ln']=np.log(r).min()
            mad['max_ln']=np.log(r).max()
            
            if self.taylor['exponent']==2:
                mad['min_ln']=mad['min_ln']-mad['params']['loc']
                mad['max_ln']=mad['max_ln']-mad['params']['loc']
                mad['params']['loc']=0

            self.mad=mad
        else:
            self.mad = mad

        if self.mad!=None:

            try:

                print(self.mad)

            except KeyError:

                print("MAD's parameters are incomplete")
        if report==True:

            if fit==True:

                return M

            else:

                self.mad_model = model

                return self.mad

    def fit_taylor(self,write=False,report=False,scale='relative',fit=True,taylor=None,tau=2):
        
        if write==False:
            print(f"Fit Taylor Law with {scale} mean and var")
            print(10*'aaaaaa')
            print(tau)
            
            x = self.observables[f'{scale} mean']['original'].values
            y = self.observables[f'{scale} var']['original'].values
            amp = FitTaylorAmplitude(x=x,y=y,steps=50,tau=tau)
            print(amp)
            self.taylor = {'exponent':tau,'amplitude':amp}

        else:

            self.taylor=taylor

        if self.taylor!={}:

            print("Taylor parameters: ")
            try:
                print("exponent",self.taylor['exponent'],"Amplitude",self.taylor['amplitude'])
            except KeyError:
                print("Taylor's parameters are incomplete")

            if report == True: 

                return taylor


class Cricket:
    
    def __init__(self,D,transforms=['relative','binary']):
        
        print('Creating a Grilli model instance')
        self.data = D
        
        self.std_transforms=transforms
        
        self.data.built_in_transform(which=transforms)
        self.observables = self.data.get_observables(zipf=True,out=True)
        self.observables = self.observables.sort_values(('zipf rank','original'))
        
    def sample_parameters(self,samples=None,mode='random',scale='original',rank_conservation=False,report=True,verbose=False):
        
        if verbose==False: end=''
        else: end=None

        print(verbose*f"Sampling {self.mad['model']} MAD",end=end)
        
        if self.mad['model']=='log-normal':
            z = np.random.normal(**self.mad['params'],size=samples)
        elif self.mad['model']=='log-laplace':
            z = np.random.laplace(**self.mad['params'],size=samples)

        m = np.exp(z)
        m = m[np.log(m)>self.mad['min_ln']]
        m = m[np.log(m)<self.mad['max_ln']]

        self.mu = np.random.choice(a=m,size=len(self.data.components))
    
    def sample_model(self,tau=2,ra_cut=-np.inf,rank_conservation=False,gamma=False):
        
        N = self.data.realization_size
        samples = self.data.samples
        nSamples = len(list(samples))
        gammaDiv = len(list(self.data.components))
        
        self.alpha = np.ones(gammaDiv)/self.taylor['amplitude']
        self.beta = self.alpha/self.mu
            
        # sample independent gammas and constrain them
            
        D=np.zeros(self.data.shape)

        for i in range(gammaDiv): 
            
            D[i,:] = np.random.gamma(shape=self.alpha[i],scale=self.beta[i]**(-1),size=nSamples)
        
        R = pd.DataFrame(columns=self.data.samples,data=np.zeros(self.data.shape))
        R.index.name = self.data.annotation

        for j,s in zip(range(nSamples),samples): 
            for i in range(gammaDiv): 
                R.loc[i,s] = np.random.poisson(lam=D[i,j]*int(N[j]))
        
        ranked_relative_abundance =(R/R.sum(axis=0)).mean(axis=1).sort_values(ascending=False)
        R=R.loc[ ranked_relative_abundance.index ]
        R.index = self.observables.index
                                
        self.sample = tb.table(R,ra_cut=ra_cut)
        
        self.sample.built_in_transform(which=self.std_transforms)
        self.sample_observables = self.sample.get_observables(zipf=True,out=True)
        self.sample_observables = self.sample_observables.sort_values(('zipf rank','original'))
                   
    def fit_mad(self,model='',scale='original',mad=None,fit=True,cut_field='loc',min_logma=-np.inf,cut=-50,ensemble=1000,report=False):
        
        r = self.observables[f'{scale} mean']['original']
        r = r[np.log(r)>min_logma]

        if fit==True:

            mad = {}
            mad['model']=model

            print(f"Fitting MAD with {model} model")

            if model=='log-normal': 
                M = ln.bootstrap_likelihood(R=r,s=ensemble)
    
            elif model=='log-laplace': 
                M = lp.bootstrap_likelihood(R=np.log(r),s=ensemble)

            M=M[M[cut_field]>cut]

            mad['params']=M.mean(axis=0).to_dict()
            mad['min_ln'], mad['max_ln'] =np.log(r).min(), np.log(r).max()

            self.mad=mad

    def fit_taylor(self,ensemble=20,report=False,scale='relative',fit=True,taylor=None,tau=2):
        
        print(f"Fit Taylor Law with {scale} mean and var")

        x = self.observables[f'{scale} mean']['original'].values
        y = self.observables[f'{scale} var']['original'].values
        amp = FitTaylorAmplitude(x=x,y=y,steps=50)
        print(amp)
        self.taylor = {'exponent':2,'amplitude':amp}

        print("Exponent",self.taylor['exponent'],"Amplitude",self.taylor['amplitude'])