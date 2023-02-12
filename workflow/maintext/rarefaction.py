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

RAREF_DIR='../../data/observables/rarefaction/'

SCH_RAREFACTION_DIR = os.path.join(RAREF_DIR,'schlomilch')
SCH_MOD_DIR = 'schlomilch'

DRC_RAREFACTION_DIR = os.path.join(RAREF_DIR,'dirichlet')
DRC_MOD_DIR = 'dirichlet'

if True:
    print(TAB_DIR)

    replicas = config['model']['replicas']

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

exps = 80

R={}

if False:

    for p in Pheno['phenotype']:
        
        print(p)

        for c in cuts:
            
            print(c)
            
            data_file=os.path.join(RAREFACTION_DIR,f'{engine}_{database}_{c}_{p}.csv')
            
            if not os.path.isfile(data_file):
            
                an.RarefactionTable( Pheno['T'][p][c],e=exps,file=data_file )
            
            for r in range(replicas):
                
                print(r)
                model_file=os.path.join(RAREFACTION_DIR,f'{MOD_DIR}/{engine}_{database}_{c}_{p}_{r}.csv')

                if not os.path.isfile(model_file):

                    MX=pd.read_csv(os.path.join(TAB_DIR,f'{c}_{p}_{r}.csv'),index_col='taxa')   
                    R=tb.table( MX )
                    R.built_in_transform(['binary'])
                    print(model_file)
                    an.RarefactionTable( R,e=exps,file=model_file,save=True )

fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

R2_score={}

title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

  
for p,j in zip(['H','U'],range(len(Pheno['phenotype']))):

    R2=pd.DataFrame(index=cuts,columns=range(replicas),dtype=float)

    for c,i in zip(cuts,range(len(cuts))):
       
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

        name=f'{engine}_{database}_{c}_{p}.csv'
        print(name)

        G=len(Pheno['T'][p][c].components)

        R=pd.read_csv(os.path.join(RAREF_DIR,name),index_col='Unnamed: 0')
        
        Xr = np.arange(1,len(R.columns)+1)
        S = len(R.columns)

        Dm=R.mean(axis=0)
        Ds=R.std(axis=0)

        ax[j,i].fill_between(x=Xr/S,y1=(Dm-Ds)/G,y2=(Dm+Ds)/G,color=col['data'][p],alpha=0.6)
        ax[j,i].plot(Xr/S,Dm/G,color=col['data'][p],linewidth=4,label=lab[p])
        
        if True:

            Dm_mod_s = pd.DataFrame(index=range(replicas),columns=R.columns)
            Ds_mod_s = pd.DataFrame(index=range(replicas),columns=R.columns)
            
            for r in range(replicas):
                
                name=f'{engine}_{database}_{c}_{p}_{r}.csv'
                R=pd.read_csv(os.path.join(SCH_RAREFACTION_DIR,name),index_col='Unnamed: 0')
            
                Gr=R.values.flatten().max()

                Dm_mod_s.loc[r]=R.mean(axis=0)/Gr
                Ds_mod_s.loc[r]=R.std(axis=0)/Gr

                ra = r2_score(Dm_mod_s.loc[r],Dm/G)
                
                R2.loc[c,r]=ra

            Dm=Dm_mod_s.mean(axis=0)
            Ds=Ds_mod_s.mean(axis=0)
            
            ax[j,i].fill_between(x=Xr/S,y1=(Dm-Ds),y2=(Dm+Ds),color=col['model'][p],alpha=0.6)
            
            ax[j,i].plot(Xr/S,Dm,color=col['model'][p],linewidth=4,label='Schlomilch')

            r2 = round(R2.loc[c].mean(),3)
            ax[j,i].text(fontsize=25,s=f'$R^2$={r2}',x=0.55,y=0.53,bbox=box )

        if True:

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


        ax[j,i].set_xlim(-0.1,1.1)
        ax[j,i].set_ylim(-0.2,1.2)

        

        if i==2:
            ax[j,i].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.525,0.175))

    R2_score[p]=R2

    ax[1,1].set_xlabel('$Fraction$ $of$ $Samples$',fontsize=45)
    ax[1,0].text(s='$Fraction$ $of$ $Species$',rotation=90,x=-0.55,y=0.65,fontsize=45)

fig.savefig(os.path.join('../../plots/maintext',f'rarefaction_{phen}_{engine}_{database}'), transparent=False, dpi=150,bbox_inches='tight')

print('done')