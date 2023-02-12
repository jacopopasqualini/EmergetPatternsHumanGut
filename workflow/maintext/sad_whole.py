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

#TAB_DIR = os.path.join('../..',config['model']['folder'],'samples')
#MOD_DIR = plot_config.model_label(config['model']['opts']['scaling_family'])

#print(TAB_DIR)

replicas = 10

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

bins = np.linspace(0,8,50)   
db=bins[1]-bins[0]
b=0.5*(bins[:-1]+bins[1:])

title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','Mid/High','High thr.$\\approx 10^{-4}$']


for c,i in zip(cuts,range(3)):
     
    for p,j in zip(Pheno['phenotype'][::-1],range(2)):

        plt_opts={'schlomilch': {'ls':'-', 'linewidth':2,'color':col['data'][p],'label':'Schlomilch'},
                  'dirichlet' : {'ls':'--','linewidth':2,'color':pl.random_rgb()},'label':'Dirichlet',
                  'Cricket' : {'ls':'--','linewidth':2,'color':pl.random_rgb()},'label':'Cricket'}


        ax[j,i].tick_params(axis='both', which='major', labelsize=40,length=25,width=4,direction='in')

        if True:
            for axis in ['bottom','left']:  
                ax[j,i].spines[axis].set_linewidth(6)

            for axis in ['right','top']:  
                ax[j,i].spines[axis].set_visible(False)

            ax[j,i].xaxis.set_major_locator(MaxNLocator(4)) 
            ax[j,i].yaxis.set_major_locator(MaxNLocator(4)) 
            
            ax[j,i].tick_params(axis='both', which='major', pad=15)

        sample = np.log10(Pheno['T'][p][c].form['original'].replace(0,np.nan)).values.flatten()
        ax[j,i].hist(sample,color=col['data'][p],alpha=0.2,density=True,histtype='stepfilled',bins=bins)
        ax[j,i].hist(sample,color=col['data'][p],density=True,histtype='step',bins=bins)
        
        M_avg=pd.DataFrame(columns=range(replicas),dtype=float)

        for g in ['schlomilch','dirichlet','Cricket']:

            for r in range(replicas):
                
                TAB_DIR = os.path.join('../../results',f'{g}RefSeqDiagnosis','samples')

                file = f'{c}_{p}_{r}.csv.zip'
                MX=pd.read_csv(os.path.join(TAB_DIR,file),index_col='taxa',compression='gzip')
                X_model = np.log10(MX.replace(0,np.nan).values.flatten())

                h=np.histogram(X_model,bins=bins)
                f=h[0]/(h[0].sum()*db)
                M_avg[r]=pd.Series(index=b,data=f)

            ax[j,i].plot(b,M_avg.mean(axis=1),**plt_opts[g])

        if j==1:
            ax[j,i].set_title(title[i],fontsize=25)
              
        ax[j,i].set_yscale('log')
        ax[j,i].set_xlim(-0.2,8.2)
        
        ax[j,i].tick_params(right=False)
        ax[j,i].tick_params(top=False)
        
        if j<1: ax[j,i].set_xticklabels([])  
        if i>0: ax[j,i].set_yticklabels([])
            
        ax[j,i].minorticks_off()

ax[0,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))
ax[1,2].legend(fontsize=25,scatterpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))

ax[1,1].set_xlabel('$Relative$ $Abundance$ $\log_{10} v $',fontsize=45)
ax[1,0].text(s="$ \mathbb{P}[ \log_{10} r. abundance$ ]$",fontsize=45,rotation=90,x=-3,y=1e-3)


fig.savefig('../../plots/maintext/sad_whole', transparent=False, dpi=150,bbox_inches='tight')