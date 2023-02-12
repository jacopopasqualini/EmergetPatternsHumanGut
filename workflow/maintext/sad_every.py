import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

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

TAB_DIR = os.path.join('../..',config['model']['folder'],'samples',f'{engine}_{database}_{phen}')
MOD_DIR = plot_config.model_label(config['model']['opts']['scaling_family'])

print(TAB_DIR)

replicas = 20#config['model']['replicas']

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


sublines = True

fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

r_bins = np.linspace(np.log10(cuts[0]),0,30)   
    
dr= r_bins[1]-r_bins[0]
b=0.5*(r_bins[:-1]+r_bins[1:])
 
title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

for c,i in zip(cuts,range(3)):
     
    for p,j in zip(['H','U'],range(2)):

        ax[j,i].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')

        for axis in ['bottom','left']:  
            ax[j,i].spines[axis].set_linewidth(6)

        for axis in ['right','top']:  
            ax[j,i].spines[axis].set_visible(False)

        ax[j,i].xaxis.set_major_locator(MaxNLocator(4)) 
        ax[j,i].yaxis.set_major_locator(MaxNLocator(4)) 
        
        ax[j,i].tick_params(axis='both', which='major', pad=15)

        samples = Pheno['T'][p][c].form['original'].sum(axis=0).sort_values().index[5:]
        sad_p = pd.DataFrame(index=b,columns=samples)
        sad_pm = pd.DataFrame(index=b,columns=samples)

        Xd,Yd=np.array([]),np.array([])
        Xm,Ym=np.array([]),np.array([])
        
        for s in samples:

            xd = Pheno['T'][p][c].form['relative'][s]
            #print(xd)
            xd = np.log10(xd[xd>0])
                
            rd = pl.ecdf4plot(xd)

            xr = np.array( rd[0] )
            yr = np.array( rd[1] )
            xr=xr[yr>0]
            yr=yr[yr>0]
            Xd=np.append(Xd,xr)
            Yd=np.append(Yd,yr)
                
            ax[j,i].scatter(xr,yr,color=col['data'][p],alpha=0.2,zorder=1)
            
         
        if i==2:
            ax[j,i].scatter([-10],[-10],color=col['data'][p],zorder=0,label=lab[p],s=150)
            
        sad_avgr = pd.Series(dtype=float)
        sad_stdr = pd.Series(dtype=float)
        
        for r in range(1):
            
            file = f'{c}_{p}_{r}.csv'
            MX=pd.read_csv(os.path.join(TAB_DIR,file),index_col='taxa')
            MX=MX/MX.sum(axis=0)

            for s in samples:
            
                xm = MX[s]
                xm = np.log10(xm[xm>0])
                
                rm = pl.ecdf4plot(xm)

                xrm = np.array( rm[0] )
                yrm = np.array( rm[1] )
                xrm = xrm[yrm>0]
                yrm = yrm[yrm>0]

                ax[j,i].scatter(xrm,yrm,color='black',alpha=0.2,zorder=2)

        if j==1:
            ax[j,i].set_title(title[i],fontsize=25)

        ax[j,i].set_yscale('log')
        ax[j,i].set_ylim(2.5e-5,5e0)
        ax[j,i].set_xlim(-6.5,0.5)
        
        ax[j,i].tick_params(right=False)
        ax[j,i].tick_params(top=False)
        
        if j<1: ax[j,i].set_xticklabels([])  
        if i>0: ax[j,i].set_yticklabels([])
            
        ax[j,i].minorticks_off()

ax[0,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))
ax[1,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))

ax[1,1].set_xlabel('$Relative$ $Abundance$ $\log_{10} v $',fontsize=45)
ax[1,0].text(s='$\mathbb{P}[$ $\log_{10} r.$ $abundance$ > $\log_{10}v ]$',fontsize=45,rotation=90,x=-10,y=1e-3)

fig.savefig(os.path.join('../../plots/maintext',f'1r_sad_{phen}_{engine}_{database}_{MOD_DIR}'), transparent=False, dpi=150,bbox_inches='tight')
