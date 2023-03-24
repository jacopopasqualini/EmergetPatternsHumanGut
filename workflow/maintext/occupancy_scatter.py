import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker

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

SCH_OBS_DIR = os.path.join('../../results','schlomilchRefSeqDiagnosis','observables')
SCH_MOD_DIR = 'schlomilch'

CRI_OBS_DIR = os.path.join('../../results','CricketRefSeqDiagnosis','observables')
CRI_MOD_DIR = 'cricket'

DRC_OBS_DIR = os.path.join('../../results','dirichletRefSeqDiagnosis','observables')
DRC_MOD_DIR = 'dirichlet'


ls = {'schlomilch':'--','cricket':'Canonical SLG','dirichlet':'-'}
model_lab = {'schlomilch':'Microc. SLG','cricket':'Canonical SLG','dirichlet':'Microc. Dirichlet'}
pheno_lab = {'H':'Healthy','U':'Unhealthy'}

if True:

    S={ cuts[0]:'Low thr.$\\approx 10^{-6}$',
        cuts[1]:'Mid thr.$\\approx 10^{-5}$',
        cuts[2]:'High thr.$\\approx 10^{-4}$' }

    replicas = 100# config['model']['replicas']

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

x_cm,y_cm = 16,8
fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(2.54*x_cm,2.54*y_cm))
fig.patch.set_facecolor('#FFFFFF')

for p,i in zip(['H','U'],range(len(Pheno['phenotype']))):

    
    R[p]=pd.DataFrame(index=cuts,columns=range(replicas))
    R[p].index=R[p].index.rename('cuts')
    
    for c,j in zip(cuts,range(len(cuts))):
        
        if True:
            ax[i,j].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in',right=False,top=False,pad=15)
            
            for axis in ['bottom','left']:  
                ax[i,j].spines[axis].set_linewidth(6)
                
            for axis in ['right','top']:  
                ax[i,j].spines[axis].set_visible(False)
            
            ax[i,j].xaxis.set_major_locator(MaxNLocator(6)) 
            ax[i,j].yaxis.set_major_locator(MaxNLocator(6)) 
        
            ax[i,j].xaxis.set_ticks(np.arange(0, 1.25, 0.25))
            ax[i,j].yaxis.set_ticks(np.arange(0, 1.25, 0.25))
            
            if j>0:  ax[i,j].set_yticklabels([])
            else:    ax[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    

        O_data = np.zeros(1)
        
        od=Pheno['X'][p][c]['binary mean']['original']
        
        for m,D in zip(['schlomilch','cricket','dirichlet'],[SCH_OBS_DIR,CRI_OBS_DIR,DRC_OBS_DIR]):

            O_avg_s = pd.DataFrame(dtype=float)

            O_model_s = np.zeros(1)

            for r in range(replicas):
            
                print(r)
                
                file = f'{c}_{p}_{r}.csv.zip'
                MX=pd.read_csv(os.path.join(D,file),index_col=0,header=[0, 1],compression='gzip')
                            
                om=MX['binary mean']['original']

                O_s=pd.DataFrame(dtype=float)
                O_s['d'],O_s['m']=od,om
                O_avg_s[r]=om

                O_s=O_s.dropna()
                
                R[p].loc[c,r]= r2_score(O_s['d'],O_s['m'])
                O_data=np.append(O_s['d'].values,O_data)
                O_model_s=np.append(O_s['m'].values,O_model_s)
                        
            O_plt_s=pd.DataFrame()
            O_plt_s['d'],O_plt_s['m']=O_s['d'],O_avg_s.mean(axis=1)
            O_plt_s=O_plt_s.dropna()
            ax[i][j].scatter(O_plt_s['d'],O_plt_s['m'],alpha=0.2,color='#848482')

            A_s=pl.binning(x=pd.Series(O_data),y=pd.Series(O_model_s),n_bins=20)
            
            ax[i,j].errorbar(x=A_s['x_mean'],xerr=A_s['x_std'],y=A_s['y_mean'],yerr=A_s['y_std'],color=col[m][p],**ebar,label= model_lab[m])
            x,y=np.random.uniform(0,1,10000),np.random.uniform(0,1,10000)

        if i==1: 
            xticks_loc = ax[i,j].get_xticks().tolist()
            ax[i,j].xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
            ax[i,j].set_xticklabels(xticks_loc,fontsize=25,fontstyle='italic')
            ax[i,j].set_title(S[c],fontsize=35)
        else: 
            ax[i,j].set_xticklabels([]) 

        if j==0:
            yticks_loc = ax[i,j].get_yticks().tolist()
            ax[i,j].yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
            ax[i,j].set_yticklabels(yticks_loc,fontsize=25,fontstyle='italic')
        else:
            ax[i,j].set_yticklabels([])        

        xi=np.linspace(-0.2,1.1)
        ax[i][j].plot(xi,xi,color='black',ls='--',linewidth=3,zorder=3)
        
        ax[i][j].set_xlim(-0.2,1.2)
        ax[i][j].set_ylim(-0.2,1.2)

ax[1,1].set_xlabel('Observed Occupancy, $[\\overline{o}_{Data}]$',fontsize=50)        
ax[1,0].text(s='Predicted Occupancy, $[\\overline{o}_{Model}]$',fontsize=50,rotation=90,x=-0.5,y=0.4)
      
ax[0,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.45,0.1))
ax[1,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.45,0.1))

fig.savefig(os.path.join('../../plots/maintext',f'occupancy_scatter_{phen}_{engine}_{database}.pdf'), transparent=False, dpi=150,bbox_inches='tight')

print('done')