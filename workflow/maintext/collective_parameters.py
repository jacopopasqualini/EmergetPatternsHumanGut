import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.ticker as ticker
import random
from sklearn.metrics import r2_score
import json

import sys

sys.path.append('../omico')
        
import plot as pl
import fit as ft
import analysis as an
import table as tb
import session as es

sys.path.append('../model')
import ecomodel as em
import laplace as lp

import plot_config as pc

CP_DIR = '../../results/collective_parameters'


def BootstrapParams(A,p=0.8,E=10):
    
    Params=pd.DataFrame(index=range(E),columns=['lambda','amplitude'])
    R=A.shape[1]
    R_tot = list(range(R))
    
    for e in range(E):
        
        random.shuffle( R_tot )

        At = tb.table(A.iloc[:,R_tot[:int(p*R)]])

        At.built_in_transform(which=['relative'])
        X_a = At.get_observables(zipf=False,out=True)

        X,Y=X_a['relative mean']['original'],X_a['relative var']['original']
        Params.loc[e,'amplitude']=em.FitTaylorAmplitude(x=X,y=Y)
        Params.loc[e,'lambda'] = lp.bootstrap_likelihood(R=np.log(X),s=50)['scale'].mean()
    return Params
    

with open(f'../../configurations/schlomilch.json') as json_file: 
    
    config = json.load(json_file)
    
data_config = config['data']

engine   = data_config['engine']
phen     = data_config['phenotype']
database = data_config['database']
protocol = data_config['protocol']
scale    = data_config['scale'] 
cuts     = data_config['cuts'] 

phen_cuts = cuts
nbins,digits=0,0
phenotype_map=pd.Series()
re_group = {'H':'H','UC':'U','CD':'U','IBS-C':'U','IBS-D':'U'}
   
Pheno = es.tables_session(engine=engine,
                          database=database,
                          PROJ_ROOT='../..',
                          cuts=phen_cuts,
                          protocol=protocol,
                          group=phen,
                          phenotype_map=phenotype_map,
                          nbins=nbins,digits=digits,
                          re_group=re_group
                          )

H_CP_file='healthy_bootstrap.csv'
if not os.path.isfile( os.path.join(CP_DIR,H_CP_file) ):

    H=Pheno['T']['H'][cuts[0]].form['original']
    Ph=BootstrapParams(A=H,E=500)
    Ph.to_csv( os.path.join(CP_DIR,H_CP_file)) 

U_CP_file='unhealthy_bootstrap.csv'
if not os.path.isfile( os.path.join(CP_DIR,U_CP_file)):

    U=Pheno['T']['U'][cuts[0]].form['original']
    Pu = BootstrapParams(A=U,E=500)
    Pu.to_csv( os.path.join(CP_DIR,U_CP_file)) 

Pu=pd.read_csv(os.path.join(CP_DIR,U_CP_file))
Ph=pd.read_csv(os.path.join(CP_DIR,H_CP_file))

#######################################################################
### PERPARE THE SCATTERS

Pu['sigma']=2*Pu['amplitude']/(1+Pu['amplitude'])
Ph['sigma']=2*Ph['amplitude']/(1+Ph['amplitude'])

Ph['lambda']=Ph['lambda']/np.sqrt(2)
Pu['lambda']=Pu['lambda']/np.sqrt(2)

Lam_h_est = round(Ph['lambda'].median(),3)
Lam_h_err = round(Ph['lambda'].std(),3)
Sig_h_est = round(Ph['sigma'].median(),3)
Sig_h_err = round(Ph['sigma'].std(),3)
H_lam='$\lambda_{H}$='+f'{Lam_h_est}'+'$\pm$'+f'{Lam_h_err}'
H_sig='$\sigma_{H}$='+f'{Sig_h_est}'+'$\pm$'+f'{Sig_h_err}'
H_est = H_sig+'\n'+H_lam

Lam_u_est = round(Pu['lambda'].median(),3)
Lam_u_err = round(Pu['lambda'].std(),3)
Sig_u_est = round(Pu['sigma'].median(),3)
Sig_u_err = round(Pu['sigma'].std(),3)
U_lam='$\lambda_{U}$='+f'{Lam_u_est}'+'$\pm$'+f'{Lam_u_err}'
U_sig='$\sigma_{U}$='+f'{Sig_u_est}'+'$\pm$'+f'{Sig_u_err}'
U_est = U_sig+'\n'+U_lam

#######################################################################
### PLOT

col= pc.phenotype_color(pheno='diagnosis')['data']

x_cm,y_cm = 8,8
fig,ax=plt.subplots(figsize=(2.54*x_cm,2.54*y_cm))
fig.patch.set_facecolor('#FFFFFF')

ax.tick_params(axis='both', which='major',labelsize=40,length=25,width=4,direction='in',pad=15,right=False,top=False)
for axis in ['bottom','left']:  ax.spines[axis].set_linewidth(4)
for axis in ['right','top']:  ax.spines[axis].set_visible(False)

ax.text(s=H_est,x=0.45,y=1.05,fontsize=40,bbox=dict(facecolor='none', edgecolor=col['H'],boxstyle='round',linewidth=5),)
plt.scatter(Ph['sigma'],Ph['lambda'],color=col['H'],alpha=0.05,s=100,zorder=2)
plt.scatter(Ph['sigma'].mean(axis=0),Ph['lambda'].mean(axis=0),color=col['H'],s=900,edgecolor='black',zorder=3,linewidth=2)

print(Ph.shape)
ax.text(s=U_est,x=1.15,y=1.05,fontsize=35,bbox=dict(facecolor='none', edgecolor=col['U'],boxstyle='round',linewidth=5))
plt.scatter(Pu['sigma'],Pu['lambda'],color=col['U'],alpha=0.05,s=100,zorder=2)
plt.scatter(Pu['sigma'].mean(axis=0),Pu['lambda'].mean(axis=0),color=col['U'],s=900,edgecolor='black',zorder=3,linewidth=2)

ax.set_xlim(0.25,1.75)
ax.set_ylim(0.875,1.125)

plt.xlabel('$Environmental$ $Fluctuations$, $[\\sigma]$',fontsize=50)
plt.ylabel('$K$ $Fluctuations$, $[\\lambda]$',fontsize=50)
ax.plot(np.ones(50),np.linspace(0.8,1.2),color='grey',ls='-',linewidth=7,zorder=1)
ax.plot(np.linspace(0.25,1.75),np.ones(50),color='grey',ls='-',linewidth=7,zorder=1)

x=np.linspace(1e-4,10,1000)

#######################################################################
healthy_pdf=True
if healthy_pdf ==True:
    ax1 = fig.add_axes([0.25,0.18,0.2,0.2])
    for axis in ['bottom','left']:  ax1.spines[axis].set_linewidth(3)
    for axis in ['right','top']:  ax1.spines[axis].set_visible(False)
    ax1.tick_params(right=False,top=False,axis='both', which='major',length=0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    sigma_h = Ph['sigma'].median()
    a_modal = (2-sigma_h)/sigma_h
    g_modal = x**(a_modal-1)*np.exp(-x)
    ax1.plot(x,g_modal,color=col['H'],linewidth=5)
    ax1.fill_between(x,np.zeros(1000),g_modal,color=col['H'],alpha=0.3)
    ax1.set_xlabel('$Species$ $Abundance$ $[v_i]$',fontsize=30)
    ax1.set_ylabel('$AFD$ $P[v_i]$',fontsize=30)

#######################################################################
unhealthy_pdf=True
if unhealthy_pdf ==True:
    ax1 = fig.add_axes([0.6,0.18,0.2,0.2])
    for axis in ['bottom','left']:  ax1.spines[axis].set_linewidth(3)
    for axis in ['right','top']:  ax1.spines[axis].set_visible(False)
    ax1.tick_params(right=False,top=False,axis='both', which='major',length=0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    sigma_h = Pu['sigma'].median()
    a_modal = (2-sigma_h)/sigma_h
    g_modal = x**(a_modal-1)*np.exp(-x)
    ax1.plot(x,g_modal,color=col['U'],linewidth=5)
    ax1.fill_between(x,np.zeros(1000),g_modal,color=col['U'],alpha=0.3)
    ax1.set_xlabel('$Species$ $Abundance$ $[v_i]$',fontsize=30)

fig.savefig(os.path.join('../../plots/maintext','collective_parameters.pdf'), transparent=False, dpi=150,bbox_inches='tight')