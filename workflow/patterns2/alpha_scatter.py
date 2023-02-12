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

TAB_DIR = os.path.join('../..',config['model']['folder'],'samples')
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


title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','midsomething','High thr.$\\approx 10^{-4}$']

R={}
fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

for p,i in zip(Pheno['phenotype'],range(len(Pheno['phenotype']))):
    
    R[p]=pd.DataFrame(index=cuts,columns=range(replicas))
    R[p].index=R[p].index.rename('cuts')
    
    for c,j in zip(cuts,range(len(cuts))):
        
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

            if j>0:  ax[i,j].set_yticklabels([])
            #else:    ax[i,j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

            if i<1:  ax[i,j].set_xticklabels([])
            #else:    ax[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    
            if i==1: ax[i,j].set_title(title[j],fontsize=25)

        A_data = np.zeros(1)
        A_model = np.zeros(1)
        
        ad=Pheno['T'][p][c].form['binary'].sum(axis=0)
        
        A_avg = pd.DataFrame(dtype=float)
        
        for r in range(replicas):
        
            #print(r)
            
            file = f'{c}_{p}_{r}.csv.zip'
            M=pd.read_csv(os.path.join(TAB_DIR,file),index_col='taxa',compression='gzip')
            print(M.shape)
            B=1*(M>0)
            am=B.sum(axis=0)
            #print(M.sum(axis=0).sort_values(),Pheno['T'][p][c].form['original'].sum(axis=0).sort_values())
            A=pd.DataFrame(dtype=float,index=ad.index)

            #A=pd.DataFrame(columns=['model','data'])
            #A['model']=am.
            #A['data']=ad
            #print(A)

            #exit()
            #print(r2_score(A['data'].sort_values(),A['model'].sort_values()))
            A['d'],A['m']=ad.sort_values().values,am.sort_values().values
            A_avg[r]=am

            A=A.dropna()
            
            R[p].loc[c,r]= r2_score(A['d'],A['m'])
            
            A_data=np.append(A['d'].values,A_data)
            A_model=np.append(A['m'].values,A_model)

            ax[i][j].scatter(A['d'],A['m'],alpha=0.2,color='#848482')
                     
        A_plt=pd.DataFrame()
        A_plt['d'],A_plt['m']=A['d'],A_avg.mean(axis=1)
        A_plt=A_plt.dropna()
        #ax[i][j].scatter(A_plt['d'],A_plt['m'],alpha=0.2,color='#848482')

        #print(A_data,A_model)
        A=pl.binning(x=pd.Series(A_data),y=pd.Series(A_model),n_bins=20)
        
        r2=round(R[p].loc[c].mean(),3)
        ax[i,j].errorbar(x=A['x_mean'],xerr=A['x_std'],y=A['y_mean'],yerr=A['y_std'],color=col['data'][p],**ebar,label=lab[p])

        Am=pd.DataFrame()
        Am['m'],Am['d']=A['x_mean'],A['y_mean']
        Am=Am.dropna()

        g=Pheno['T'][p][c].shape[0]

        xi=np.linspace(0,g)
        ax[i][j].plot(xi,xi,color='black',ls='--',linewidth=3,zorder=3)
        
        ax[i][j].set_xlim(0,g)
        ax[i][j].set_ylim(0,g)

        exit()
        
        #ax[i,j].text(fontsize=25,s=f'$R^2$={r2}',x=0.6*g,y=0.25*g,bbox=box )
        
ax[0,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc='upper left')
ax[1,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc='upper left')

OUTDIR=os.path.join(config['model']['folder'])

PLOT_DIR=os.path.join(OUTDIR,f'plots/')
es.check_path(PLOT_DIR)

fig.savefig(os.path.join('../..',PLOT_DIR,'alpha_scatter'), transparent=False, dpi=150,bbox_inches='tight')


Rtot= pd.DataFrame( columns = pd.MultiIndex.from_tuples([(p,r) for p in Pheno['phenotype'] for r in range(replicas)] ) )

for p in Pheno['phenotype']: Rtot[p]=R[p]

TEST_DIR=os.path.join(OUTDIR,f'test/')
es.check_path(TEST_DIR)

Rtot.to_csv(os.path.join('../..',TEST_DIR,'R2_alpha.csv'),index=True)


print('done')