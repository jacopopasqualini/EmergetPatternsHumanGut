import pandas as pd
import numpy as np
import scipy as sp
import random
from scipy import stats
from scipy.optimize import fsolve

def truncatedLaplace(D,epsilon=1e-5):
    
    c = D.min()
    
    def L(p):
        
        # parse th esingle parmeters
        mu, la = p
        
        # introduce empirical moments
        m1 = np.mean(D)
        m2 = np.var(D)
        
        phi = np.exp( (c-mu)/la)
        
        M = (mu-0.5*phi*(c-la))/(1-0.5*phi) - m1
        
        S = ( ( (phi/2)**2 + 2*(1-phi) )*la**2 -(phi/2)*(c-mu)*(c-mu-2*la)  )/(1-0.5*phi)**2 - m2
        
        return (M,S)
    
    return L

def bootstrap_likelihood(R,freqs=None,bins=None,chi=False,s=100,p=0.75):

    r = R.shape[0]

    if chi==True:  fields = ['loc','scale','chi']
    else:          fields = ['loc','scale']
        
    Z=pd.DataFrame(index=range(s),columns=fields)
        
    for i in range(s):
        
        x = np.random.choice(a=R,size=int(p*r))
        
        P=fsolve(truncatedLaplace(x), ( np.mean(x), np.var(x)) )
        Z.loc[i,'loc'],Z.loc[i,'scale'] = P[0],P[1]

    return Z