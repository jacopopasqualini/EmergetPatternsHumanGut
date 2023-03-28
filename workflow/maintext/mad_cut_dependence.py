import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.ticker as ticker

from sklearn.metrics import r2_score
import json

import sys

sys.path.append('../omico')
        
import plot as pl
import fit as ft
import analysis as an
import table as tb
import session as es

import plot_config

sys.path.append('../model')
import ecomodel as em
import lognormal as ln
import laplace as lp

################################################################
def BIC(pdf,k,data):
    
    lnL = 2 * np.log(pdf).sum()
    
    return k*np.log( data.shape[0]) - lnL

################################################################
config_file = sys.argv[1]

with open(f'../../configurations/{config_file}.json') as json_file: 
    
    config = json.load(json_file)

data_config = config['data']

engine   = data_config['engine']
phen     = data_config['phenotype']
database = data_config['database']
protocol = data_config['protocol']
scale    = data_config['scale'] 
cuts     = data_config['cuts'] 

cuts=np.logspace(np.log10(cuts[0]),np.log10(cuts[-1]),50)
nbins,digits=0,0
phenotype_map=pd.Series()
re_group = {'H':'H','UC':'U','CD':'U','IBS-C':'U','IBS-D':'U'}

lab  = plot_config.phenotype_legend(phen)
col  = plot_config.phenotype_color(phen)
box  = plot_config.box_style()
ebar = plot_config.ebar_style()
   
Pheno = es.tables_session(engine=engine,
                          database=database,
                          PROJ_ROOT='../..',
                          cuts=cuts,
                          protocol=protocol,
                          group=phen,
                          phenotype_map=phenotype_map,
                          nbins=nbins,digits=digits,
                          re_group=re_group
                          )

Plot = False

if Plot:
    fig,ax=plt.subplots(nrows=len(cuts),ncols=len(Pheno['phenotype']),figsize=(15,60))

R_A_normal=pd.DataFrame(index=cuts,columns=['mean','sigma','BIC'])
R_A_laplace=pd.DataFrame(index=cuts,columns=['mu','lambda','BIC'])

BIC_normal=pd.DataFrame(index=cuts,columns=Pheno['phenotype'])
BIC_laplace=pd.DataFrame(index=cuts,columns=Pheno['phenotype'])

i=0

for c,i in zip(cuts,range(len(cuts))):
    
    for p,j in zip(Pheno['phenotype'],range(len(Pheno['phenotype']))):
    
        M = np.log(Pheno['X'][p][c]['relative mean']['original'])
        X=np.linspace(M.min(),M.max(),200)
        
        ###########################################################################
        ### LOG-NORMAL MAD
        
        Gp = em.Cricket(D=Pheno['T'][p][c])
        Gp.fit_taylor(write=True,taylor={'exponent':2,'A':1})
        Gp.fit_mad(scale='relative',report=True,ensemble=50,cut=-50,model='log-normal',cut_field='loc')
        nor = Gp.mad['params']
        
        normal  = stats.norm.pdf(X,loc=nor['loc'] ,scale=nor['scale'] )
        Zn = 1-ln.gaussian_cumulative((M.min()-nor['loc'])/nor['scale'])
 
        del Gp
        
        ###########################################################################
        ### LOG-LAPLACE MAD
        
        Gp = em.Cricket(D=Pheno['T'][p][c])
        Gp.fit_taylor(write=True,taylor={'exponent':2,'A':1})
        Gp.fit_mad(scale='relative',report=True,ensemble=50,cut=-50,model='log-laplace',cut_field='loc')
        lap = Gp.mad['params']
        
        laplace  = stats.laplace.pdf(X,loc=lap['loc'],scale=lap['scale'] )
        Zl = 1 - 0.5*np.exp((M.min()-lap['loc'])/lap['scale'])
        
        del Gp
        
        ###########################################################################
        ### EVALUATE BIC
        
        normal  = stats.norm.pdf(M,loc=nor['loc'],scale=nor['scale'])/Zn
        BIC_normal.loc[c,p]=BIC(pdf=normal,k=2,data=M)

        laplace  = stats.laplace.pdf(M,loc=lap['loc'],scale=lap['scale'] )/Zl
        BIC_laplace.loc[c,p]=BIC(pdf=laplace,k=2,data=M)

x_cm,y_cm = 14,8

fig, ax = plt.subplots(figsize=(2.54*x_cm,2.54*y_cm))
fig.patch.set_facecolor('#FFFFFF')
    
ax.tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in',right=False,top=False,pad=15)
for axis in ['bottom','left']:  ax.spines[axis].set_linewidth(6)
for axis in ['right','top']: ax.spines[axis].set_visible(False)

ax.tick_params(which='minor', width=2,length=25/2.5,top=False)

fig.patch.set_facecolor('#FFFFFF')

BIC_ratio = BIC_normal/BIC_laplace

ax.plot(cuts,BIC_ratio['H'],linewidth=5.5,color=col['data']['H'],label='Helathy')
ax.plot(cuts,BIC_ratio['U'],linewidth=5.5,color=col['data']['U'],label='Unhealthy')

plt.plot(np.logspace(-9.5,-3.5),np.ones(50),linewidth=5.5,color='black',ls='--')

ax.legend(fontsize=35 ,fancybox=True,shadow=True,loc='upper right')

ax.set_xscale('log')
ax.set_xlim(10**(-6.5),10**(-3.5))
ax.set_xlabel('$Relative$ $abundance$ $cut$-$off$, $\\kappa$',fontsize=50)

ax.set_ylim(0.65,1.25)
ax.set_ylabel('$Normal$-$Laplace$ $BIC$ $ratio$',fontsize=50)

############################################################################
low_cut_plot = True

if low_cut_plot:
    w=0.3
    ax2 = fig.add_axes([0.174,0.2,w,w])
    ax2.axis('off')
    v=np.log(Pheno['X']['U'][cuts[0]]['relative mean']['original'])
    v=(v-v.mean())/v.std()
    ax2.hist(v, bins=50,histtype='stepfilled',density=True,color='#BFBFBC',zorder=0)
    ax2.hist(v, bins=50,histtype='step',density=True,color='black',zorder=0,linewidth=2)
    ab = np.linspace(v.min(),v.max(),num=200)
    normal  = stats.norm.pdf(ab, loc=0,scale=1)
    laplace = stats.laplace.pdf(ab,loc=0,scale=2**(-0.5))
    ax2.plot(ab,normal,color='#88E603',linewidth=4,zorder=1)
    ax2.plot(ab,laplace,linewidth=4,zorder=0,color='#FF5500')
    ax2.text(s='$Log$ $Mean$ $Abundance$',fontsize=37.5,x=-1.5,y=-0.1)
    ax2.text(s='$Frequency$',fontsize=37.5,rotation=90,x=-3.6,y=0.2)
    ax2.set_xlim(-3.25,3.25)

###############################

high_cut_plot = True
if high_cut_plot:
    ax3 = fig.add_axes([0.55,0.2,w,w])
    ax3.axis('off')
    v=np.log(Pheno['X']['U'][cuts[int(0.55*len(cuts))]]['relative mean']['original'])
    v=(v-v.mean())/v.std()
    ax3.hist(v, bins=50,histtype='stepfilled',density=True,color='#BFBFBC',zorder=0)
    ax3.hist(v, bins=50,histtype='step',density=True,color='black',zorder=0,linewidth=2)
    ab = np.linspace(v.min(),v.max(),num=200)
    normal  = stats.norm.pdf(ab, loc=0,scale=1)
    laplace = stats.laplace.pdf(ab,loc=0,scale=2**(-0.5))
    ax3.plot(ab,normal,color='#88E603',linewidth=4,zorder=1,label='Normal')
    ax3.plot(ab,laplace,linewidth=4,zorder=0,color='#FF5500',label='Laplace')
    ax3.text(s='$Log$ $Mean$ $Abundance$',fontsize=37.5,x=-1.225,y=-0.1)
    ax3.legend(fontsize=35 ,fancybox=True,shadow=True,loc='upper right')
    ax3.set_xlim(-3.25,3.25)

fig.savefig(os.path.join('../../plots/maintext','mad_cut_dependence.pdf'), transparent=True, dpi=150,bbox_inches='tight')
