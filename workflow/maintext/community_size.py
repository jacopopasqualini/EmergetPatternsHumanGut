import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
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

RES_DIR='../../results'

SCH_DIR = os.path.join(RES_DIR,'schlomilchRefSeqDiagnosis/samples')
GRI_DIR = os.path.join(RES_DIR,'CricketRefSeqDiagnosis/samples')
DRC_DIR = os.path.join(RES_DIR,'dirichletRefSeqDiagnosis/samples')

if True:

    replicas = 5# config['model']['replicas']

    lab   = plot_config.phenotype_legend(phen)
    col   = plot_config.phenotype_color(phen)
    box   = plot_config.box_style()
    ebar  = plot_config.ebar_style()

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

fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

R2_score={}

title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

  
for p,j in zip(['H','U'],range(len(Pheno['phenotype']))):

    R2=pd.DataFrame(index=cuts,columns=range(replicas),dtype=float)

    for c,i in zip(cuts,range(len(cuts))):
       
        # making this plot nice
        if True:
            ax[j,i].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')

            for axis in ['bottom','left']:  
                ax[j,i].spines[axis].set_linewidth(6)

            for axis in ['right','top']:  
                ax[j,i].spines[axis].set_visible(False)

            #ax[j,i].xaxis.set_major_locator(MaxNLocator(5)) 
            #ax[j,i].yaxis.set_major_locator(MaxNLocator(4)) 

            ax[j,i].tick_params(right=False)
            ax[j,i].tick_params(top=False)
            ax[j,i].tick_params(axis='both', which='major', pad=15)

            ax[j,i].xaxis.set_ticks(np.arange(0, 1.25, 0.25))
            ax[j,i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

            if j<1: 
                ax[j,i].set_xticklabels([])
            else:
                ax[j,i].set_xticklabels(ax[j,i].get_xticks(), rotation = 45)

            ax[j,i].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

            if i>0: 
                ax[j,i].set_yticklabels([])
            else:
                ax[j,i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    
            if j==1:
                ax[j,i].set_title(title[i],fontsize=25)

        N_data = Pheno['T'][p][c].realization_size.sort_values().values

        ax[j,i].plot((N_data),(N_data),color='black')
        for r in range(replicas):

            name=f'{c}_{p}_{r}.csv.zip'
            
            M_gri=pd.read_csv(os.path.join(GRI_DIR,name),index_col='taxa',compression='gzip')
            N_gri = M_gri.sum(axis=0).sort_values().values
            ax[j,i].scatter((N_data),(N_gri),color='blue')

            M_sch=pd.read_csv(os.path.join(SCH_DIR,name),index_col='taxa',compression='gzip')
            N_sch = M_sch.sum(axis=0).sort_values().values
            ax[j,i].scatter((N_data),(N_sch),color='red')


        if False:

            Dm_mod_d = pd.DataFrame(index=range(replicas),columns=R.columns)
            Ds_mod_d = pd.DataFrame(index=range(replicas),columns=R.columns)
            
            for r in range(replicas):
                
                name=f'{engine}_{database}_{c}_{p}_{r}.csv'
                R=pd.read_csv(os.path.join(DRC_RAREFACTION_DIR,name),index_col='Unnamed: 0')
            
                Gr=R.values.flatten().max()

                Dm_mod_d.loc[r]=R.mean(axis=0)/Gr
                Ds_mod_d.loc[r]=R.std(axis=0)/Gr
                

            Dm=Dm_mod_d.mean(axis=0)
            Ds=Ds_mod_d.mean(axis=0)
                        
            ax[j,i].plot(Xr/S,Dm,linewidth=4,color='black',ls='--',label='Dirichlet')


        #ax[j,i].set_xlim(-0.1,1.1)
        #ax[j,i].set_ylim(-0.2,1.2)

        

        #if i==2:
            ax[j,i].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.525,0.175))

    #R2_score[p]=R2

    #ax[1,1].set_xlabel('$Fraction$ $of$ $Samples$',fontsize=45)
    #ax[1,0].text(s='$Fraction$ $of$ $Species$',rotation=90,x=-0.55,y=0.65,fontsize=45)

fig.savefig(os.path.join('../../plots/maintext',f'community_siize'), transparent=False, dpi=150,bbox_inches='tight')

print('done')