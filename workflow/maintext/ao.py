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

SCH_OBS_DIR = os.path.join('../../results','schlomilchRefSeqDiagnosis','observables')
SCH_MOD_DIR = 'schlomilch'

CRI_OBS_DIR = os.path.join('../../results','CricketRefSeqDiagnosis','observables')
CRI_MOD_DIR = 'Cricket'

DRC_OBS_DIR = os.path.join('../../results','dirichletRefSeqDiagnosis','observables')
DRC_MOD_DIR = 'dirichlet'

ls = {'schlomilch':'--','cricket':'Canonical SLG','dirichlet':'-'}
model_lab = {'schlomilch':'Microc. SLG','cricket':'Canonical SLG','dirichlet':'Microc. Dirichlet'}
pheno_lab = {'H':'Healthy','U':'Unhealthy'}
title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']


if True:

    #print('>>>>>>>>>>>>>',MOD_DIR)
    replicas = 100

    lab  = plot_config.phenotype_legend(phen)
    col  = plot_config.phenotype_color(phen)
    box  = plot_config.box_style()
    ebar = plot_config.ebar_style()

    phen_cuts=cuts
    nbins,digits=0,0
    phenotype_map=pd.Series(dtype='string')
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

x_cm,y_cm = 16,8
fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(2.54*x_cm,2.54*y_cm))
fig.patch.set_facecolor('#FFFFFF')

R={}


for p,i in zip(['H','U'],range(len(Pheno['phenotype']))):
    
    for c,j in zip(cuts,range(len(cuts))):
        
        if True:
            ax[i,j].tick_params(axis='both', which='major',top=False,right=False,labelsize=40,length=25,width=4,direction='in',pad=15)
            ax[i,j].tick_params(axis='x', which='minor', top=False,width=2,length=25/2.5)
            for axis in ['bottom','left']:  ax[i,j].spines[axis].set_linewidth(6)
            for axis in ['right','top']: ax[i,j].spines[axis].set_visible(False)
            
            #ax[i,j].xaxis.set_major_locator(MaxNLocator(5)) 
            ax[i,j].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

            if j>0:  ax[i,j].set_yticklabels([])
            else:    ax[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    
            if i==1: ax[i,j].set_title(title[j],fontsize=35)

        md=Pheno['X'][p][c]['relative mean']['original']
        od=Pheno['X'][p][c]['binary mean']['original']
        #md,od=np.log10(md),od
        
        AO_data=pd.DataFrame()
        AO_data['x'],AO_data['y']=md,od
        AO_data=AO_data.dropna()
        
        ax[i][j].scatter(AO_data['x'],AO_data['y'],alpha=0.2,color='#848482',zorder=0)
        P=pl.binning(x=AO_data['x'],y=AO_data['y'],n_bins=15,scale='log' )

        if j == 0: d_label = pheno_lab[p]

        ax[i][j].errorbar(x=P['x_mean'],xerr=P['x_std'],y=P['y_mean'],yerr=P['y_std'],color=col['data'][p],**ebar,zorder=3,ls='none',label=d_label)

        

        for m,D in zip(['schlomilch','cricket','dirichlet'],[SCH_OBS_DIR,CRI_OBS_DIR,DRC_OBS_DIR]):

            M_model_s,O_model_s=np.zeros(1),np.zeros(1)

            for r in range(replicas):

                file = f'{c}_{p}_{r}.csv.zip'
                MX=pd.read_csv(os.path.join(D,file),index_col=0,header=[0, 1],compression='gzip')
                    
                mm = MX['relative mean']['original'] 
                om = MX['binary mean']['original']
            
                M_model_s,O_model_s=np.append(M_model_s,mm),np.append(O_model_s,om)
                
                AO_model_s=pd.DataFrame()
                AO_model_s['x'],AO_model_s['y']=mm,om
                AO_model_s=AO_model_s.dropna()
                #AO_model_s=AO_model_s[AO_model_s['x']<0]

            P_s=pl.binning(x=AO_model_s['x'],y=AO_model_s['y'],n_bins=40,scale='log')

            ax[i][j].plot(P_s['x_mean'],P_s['y_mean'],color=col[m][p],zorder=2,linewidth=5.5,label= model_lab[m])


        ax[i][j].set_xlim(10**(-9.2),10**(0.2))
        ax[i][j].set_xscale('log')
        ax[i][j].set_ylim(-0.2,1.2)
        
        if j==2:
            ax[i,j].legend(fontsize=22.5,numpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.5125))
        
ax[1,1].set_xlabel('$Mean$ $Relative$ $Abundance$, $[\overline{\\mu}]$',fontsize=50)
ax[1,0].text(s='$Occupancy$, $[\overline{o}]$',rotation=90,x=10**(-11.5),y=1,fontsize=50)

fig.savefig(os.path.join('../../plots/maintext',f'ao_{phen}_{engine}_{database}_comparison.pdf'), transparent=False, dpi=150,bbox_inches='tight')

OUTDIR='../../data/observables/ao'

print('done')