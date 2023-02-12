import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker

import json

import sys

sys.path.append('../omico')
        
import plot as pl
import fit as ft
import analysis as an
import table as tb
import session as es

import plot_config

##############################
# LOAD DATASET

config_file = sys.argv[1]

with open(f'../../data/configurations/{config_file}.json') as json_file: 
    
    config = json.load(json_file)

data_config = config['data']

engine   = data_config['engine']
phen     = data_config['phenotype']
database = data_config['database']
protocol = data_config['protocol']
scale    = data_config['scale'] 
cuts     = data_config['cuts']

SCH_OBS_DIR = os.path.join('../../data/model','schlomilch','observables',f'{engine}_{database}_{phen}')
SCH_MOD_DIR = 'schlomilch'

DRC_OBS_DIR = os.path.join('../../data/model','dirichlet','observables',f'{engine}_{database}_{phen}')
DRC_MOD_DIR = 'dirichlet'

if True:

    S={ cuts[0]:'Low thr.$\\approx 10^{-6}$',
        cuts[1]:'Mid thr.$\\approx 10^{-5}$',
        cuts[2]:'High thr.$\\approx 10^{-4}$' }

    replicas = config['model']['replicas']

    lab  = plot_config.phenotype_legend(phen)
    col  = plot_config.phenotype_color(phen)
    box  = plot_config.box_style()
    ebar = plot_config.ebar_style()

    phen_cuts=cuts
    nbins,digits=0,0
    phenotype_map=pd.Series()
    re_group={}

    if phen=='community_size':
        
        phenotype_map = np.log10( Complete['T']['all'][cuts[0]].realization_size )
        nbins,digits=4,3
        phen_cuts=[ cuts[0],cuts[1] ]
        cuts=phen_cuts
        
    elif phen=='age': nbins,digits=5,0
    elif phen=='bmi': nbins,digits=5,1
    elif phen=='diagnosis': re_group = {'H':'H','UC':'U','CD':'U','IBS-C':'U','IBS-D':'U'}
    else: pass

    
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



R={}
fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

S=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

lab={'H':'Healthy','U':'Disease'}

col={'H':'#72A0C1','U':'#B81D27'}


fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

left = [-7,-6.5,-5.5]
for p,i in zip(['H','U'],range(2)):
    
    print(p)
    
    for c,j in zip(cuts,range(3)):
        
        if True:

            ax[i,j].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')
            
            for axis in ['bottom','left']:  
                ax[i,j].spines[axis].set_linewidth(6)
                
            for axis in ['right','top']:  
                ax[i,j].spines[axis].set_visible(False)
            
            ax[i,j].xaxis.set_major_locator(MaxNLocator(5)) 
            ax[i,j].yaxis.set_major_locator(MaxNLocator(5)) 
        
            ax[i,j].tick_params(right=False)
            ax[i,j].tick_params(top=False)
            ax[i,j].tick_params(axis='both', which='major', pad=15)

        x=np.log10( Pheno['X'][p][c]['relative mean']['original'] )
        y=np.log10( Pheno['X'][p][c]['relative var']['original'] )
        ax[i,j].scatter(x,y,alpha=0.6,zorder=0,s=150,color=col[p],label=lab[p])
     
        #x=np.log10( Pheno['X'][p][c]['relative mean']['original'] )
        #y=np.log10( Pheno['X'][p][c]['relative var']['original'] )
        #ax[i,j].scatter(x,y,alpha=0.6,zorder=0,s=150,color=col[p])
                   
        xm = min(x.min(),x.min())
        xM = min(x.max(),x.max())
        ym = min(y.min(),y.min())

        #xi = np.linspace(left[j],xM)
        #A = y.mean()-2*x.mean()
        #ax[i,j].plot(xi,2*xi+A,color=pl.complementary_rgb(col[p]),linewidth=7,zorder=2)

        x=np.linspace(xm,xM-1)

        ax[i,j].plot(x,ym+x-xm,color='black',linewidth=7,ls='--',alpha=0.7,zorder=0)
        ax[i,j].plot(x,ym+2*(x-xm),color='black',linewidth=7,ls='--',alpha=0.7,zorder=0)

        ax[i,j].set_xlim(-9.5,-0.5)
        ax[i,j].set_ylim(-16.5,-0.5)
         

ax[1,1].set_xlabel('$Mean$ $Relative$ $Abundance$ $x=\log_{10} \overline{v}$',fontsize=45)
ax[1,0].text(s='$Relative$ $Variance$ $y=\log_{10} \\sigma^2_{v}$',fontsize=45,rotation=90,x=-13,y=-9)

ax[0,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.5,0.2))
ax[1,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.5,0.2))

ax[0,0].text(s='$y=x^2$',x=-6.5,y=-8.75,fontsize=30,rotation=45)
ax[0,0].text(s='$y=x$',x=-5.5,y=-12.5,fontsize=30,rotation=22.5)

for i,t,m in zip(range(3),['Low thr.','Mid thr.','High thr.'],['6','5','4']):
    
    ax[1,i].set_title(S[i],fontsize=25)
    
fig.savefig(os.path.join('../../plots/maintext','taylor_open'), transparent=False, dpi=150,bbox_inches='tight')