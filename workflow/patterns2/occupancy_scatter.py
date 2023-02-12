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

with open(f'../../configurations/{config_file}.json') as json_file: 
    
    config = json.load(json_file)

data_config = config['data']

engine   = data_config['engine']
phen     = data_config['phenotype']
database = data_config['database']
protocol = data_config['protocol']
scale    = data_config['scale'] 
cuts     = data_config['cuts']

OBS_DIR = os.path.join('../..',config['model']['folder'],'observables')
MOD_DIR = plot_config.model_label(config['model']['opts']['scaling_family'])

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

for p,i in zip(Pheno['phenotype'],range(len(Pheno['phenotype']))):

    
    R[p]=pd.DataFrame(index=cuts,columns=range(replicas))
    R[p].index=R[p].index.rename('cuts')
    
    for c,j in zip(cuts,range(len(cuts))):
        
        ax[i,j].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')
        
        for axis in ['bottom','left']:  
            ax[i,j].spines[axis].set_linewidth(6)
            
        for axis in ['right','top']:  
            ax[i,j].spines[axis].set_visible(False)
        
        ax[i,j].xaxis.set_major_locator(MaxNLocator(6)) 
        ax[i,j].yaxis.set_major_locator(MaxNLocator(6)) 
    
        ax[i,j].tick_params(right=False)
        ax[i,j].tick_params(top=False)
        ax[i,j].tick_params(axis='both', which='major', pad=15)

        O_data = np.zeros(1)
        O_model = np.zeros(1)
        
        od=Pheno['X'][p][c]['binary mean']['original']
        
        O_avg = pd.DataFrame(dtype=float)
        
        for r in range(replicas):
        
            print(r)
            
            file = f'{c}_{p}_{r}.csv.zip'
            MX=pd.read_csv(os.path.join(OBS_DIR,file),index_col=0,header=[0, 1],compression='gzip')
                           
            om=MX['binary mean']['original']

            O=pd.DataFrame(dtype=float)
            O['d'],O['m']=od,om
            O_avg[r]=om

            O=O.dropna()
            
            R[p].loc[c,r]= r2_score(O['d'],O['m'])
            
            O_data=np.append(O['d'].values,O_data)
            O_model=np.append(O['m'].values,O_model)
                     
        O_plt=pd.DataFrame()
        O_plt['d'],O_plt['m']=O['d'],O_avg.mean(axis=1)
        O_plt=O_plt.dropna()
        ax[i][j].scatter(O_plt['d'],O_plt['m'],alpha=0.2,color='#848482')

        A=pl.binning(x=pd.Series(O_data),y=pd.Series(O_model),n_bins=30)
        
        r2=round(R[p].loc[c].mean(),3)
        ax[i,j].errorbar(x=A['x_mean'],xerr=A['x_std'],y=A['y_mean'],yerr=A['y_std'],color=col['data'][p],**ebar,label=lab[p])

        Am=pd.DataFrame()
        Am['m'],Am['d']=A['x_mean'],A['y_mean']
        Am=Am.dropna()

        xi=np.linspace(-0.1,1.1)
        ax[i][j].plot(xi,xi,color='black',ls='--',linewidth=3,zorder=3)
        
        ax[i][j].set_xlim(-0.2,1.2)
        ax[i][j].set_ylim(-0.2,1.2)
        
        ax[i,j].text(fontsize=25,s=f'$R^2$={r2}',x=0.6,y=0,bbox=box )
        
ax[0,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc='upper left')
ax[1,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc='upper left')

OUTDIR=os.path.join(config['model']['folder'])

PLOT_DIR=os.path.join(OUTDIR,f'plots/')
es.check_path(PLOT_DIR)

fig.savefig(os.path.join('../..',PLOT_DIR,'occupancy_scatter'), transparent=False, dpi=150,bbox_inches='tight')

print(R)
Rtot= pd.DataFrame( columns = pd.MultiIndex.from_tuples([(p,r) for p in Pheno['phenotype'] for r in range(replicas)] ) )

for p in Pheno['phenotype']: print(R[p])

for p in Pheno['phenotype']: Rtot[p]=R[p]

TEST_DIR=os.path.join(OUTDIR,f'test/')
es.check_path(TEST_DIR)

Rtot.to_csv(os.path.join('../..',TEST_DIR,'R2_occupancy.csv'),index=True)

print('done')