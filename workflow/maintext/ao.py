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

    #print('>>>>>>>>>>>>>',MOD_DIR)
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

fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

R={}

title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

for p,i in zip(['H','U'],range(len(Pheno['phenotype']))):

    R2=pd.DataFrame(index=cuts,columns=range(replicas),dtype=float)
    
    for c,j in zip(cuts,range(len(cuts))):
        
        if True:
            ax[i,j].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')
            
            for axis in ['bottom','left']:  
                ax[i,j].spines[axis].set_linewidth(6)
                
            for axis in ['right','top']:  
                ax[i,j].spines[axis].set_visible(False)
            
            ax[i,j].xaxis.set_major_locator(MaxNLocator(5)) 
        
            ax[i,j].tick_params(right=False)
            ax[i,j].tick_params(top=False)
            ax[i,j].tick_params(axis='both', which='major', pad=15)

            ax[i,j].yaxis.set_ticks(np.arange(0, 1.25, 0.25))

            if j>0:  ax[i,j].set_yticklabels([])
            else:    ax[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    
            if i==1: ax[i,j].set_title(title[j],fontsize=25)

        md=Pheno['X'][p][c]['relative mean']['original']
        od=Pheno['X'][p][c]['binary mean']['original']
        md,od=np.log10(md),od
        
        AO_data=pd.DataFrame()
        AO_data['x'],AO_data['y']=md,od
        AO_data=AO_data.dropna()
        
        ax[i][j].scatter(AO_data['x'],AO_data['y'],alpha=0.2,color='#848482',zorder=0)
        P=pl.binning(x=AO_data['x'],y=AO_data['y'],n_bins=10)
        ax[i][j].errorbar(x=P['x_mean'],xerr=P['x_std'],y=P['y_mean'],yerr=P['y_std'],color=col['data'][p],**ebar,zorder=3,label=lab[p])
    
        if True:

            M_model_s,O_model_s=np.zeros(1),np.zeros(1)

            for r in range(replicas):

                file = f'{c}_{p}_{r}.csv'
                MX=pd.read_csv(os.path.join(SCH_OBS_DIR,file),index_col=0,header=[0, 1])
                    
                mm=np.log10( MX['relative mean']['original'] )
                om=MX['binary mean']['original']
            
                M_model_s,O_model_s=np.append(M_model_s,mm),np.append(O_model_s,om)
                
                AO_model_s=pd.DataFrame()
                AO_model_s['x'],AO_model_s['y']=mm,om
                AO_model_s=AO_model_s.dropna()
                AO_model_s=AO_model_s[AO_model_s['x']<0]

            R2.loc[c,r]=an.compare_neutral_curves(AO_data,AO_model_s,delta=0.2,n_bins=15)
            P_s=pl.binning(x=AO_model_s['x'],y=AO_model_s['y'],n_bins=40)
        
            yl_s=np.array(P_s['y_mean'])-3*np.array(P_s['y_std'])
            yu_s=np.array(P_s['y_mean'])+3*np.array(P_s['y_std'])
            
            ax[i][j].fill_between(x=np.array(P_s['x_mean']),y1=yl_s,y2=yu_s,color=col['model'][p],alpha=0.5,zorder=1)
            ax[i][j].plot(P_s['x_mean'],P_s['y_mean'],color=col['model'][p],zorder=2,linewidth=5.5,label='Schlomich')#ax[i][0].set_ylabel('Model')
            
            r2=round(R2.loc[c].mean(),3)
            ax[i,j].text(fontsize=25,s='$R_{Sch.}^2$='+f'{r2}',x=-3.5,y=0,bbox=box )
        
        if True:

            M_model_d,O_model_d=np.zeros(1),np.zeros(1)

            for r in range(replicas):

                file = f'{c}_{p}_{r}.csv'
                MX=pd.read_csv(os.path.join(DRC_OBS_DIR,file),index_col=0,header=[0, 1])
                    
                mm=np.log10( MX['relative mean']['original'] )
                om=MX['binary mean']['original']
            
                M_model_d,O_model_d=np.append(M_model_d,mm),np.append(O_model_d,om)
                
                AO_model_d=pd.DataFrame()
                AO_model_d['x'],AO_model_d['y']=mm,om
                AO_model_d=AO_model_d.dropna()
                AO_model_d=AO_model_d[AO_model_d['x']<0]

            P_d=pl.binning(x=AO_model_d['x'],y=AO_model_d['y'],n_bins=40)
        
            ax[i][j].plot(P_d['x_mean'],P_d['y_mean'],color='black',zorder=2,linewidth=5.5,ls='--',label='Dirichlet')
            
        ax[i][j].set_xlim(-9,0)
        ax[i][j].set_ylim(-0.2,1.2)
        
        if j==0:
            ax[i,j].legend(fontsize=22.5,numpoints=1 ,fancybox=True,shadow=True,loc=(0.5125,0.275))

        if j==2:
            ax[i,j].yaxis.set_label_position("right")
            ax[i,j].set_ylabel(lab[p],rotation=270,labelpad=50,fontsize=45)

        
    R[p]=R2

ax[1,1].set_xlabel('$Mean$ $Relative$ $Abundance$, $\log_{10} \overline{\\mu}$',fontsize=45)
ax[1,0].text(s='$Occupancy$, $\overline{o}$',rotation=90,x=-13,y=0.65,fontsize=45)

fig.savefig(os.path.join('../../plots/maintext',f'ao_{phen}_{engine}_{database}_comparison'), transparent=False, dpi=150,bbox_inches='tight')

OUTDIR='../../data/observables/ao'

print('done')