import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.ticker as ticker

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

replicas = 3#config['model']['replicas']

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


title=['Low thr.$\\approx 10^{-6}$','kkifd','Mid thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

R={}
fig,ax=plt.subplots(nrows=len(Pheno['phenotype']),ncols=len(cuts),figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

cb = {'H':'Blues','U':'Reds'}

bins = np.linspace(-1,1,20)
db=bins[1]-bins[0]
b=0.5*(bins[:-1]+bins[1:])

for p,i in zip(['H','U'],range(len(Pheno['phenotype']))):
    
    R[p]=pd.DataFrame(index=cuts,columns=list(range(replicas)))
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
        
        D=Pheno['T'][p][c].form['relative']
        #D=np.log10( D.replace(0,np.nan) ).fillna(0)

        pearson_data=np.triu(D.T.corr().values, k=1).flatten()
        #pearson_data=pearson_data[pearson_data!=0]
        #pearson_data=pearson_data[pearson_data!=1]

        h=np.histogram(pearson_data,bins=bins)
        f=h[0]/(h[0].sum()*db)
        qk=f.copy()
        w=np.where(f>0)
        F_model = pd.Series(index=b,data=f)
        bx,f=b[w],f[w]

        ax[i][j].plot(bx,f,color=col['data'][p],linewidth=4,label=lab[p])

        P_avg=pd.DataFrame(columns=range(replicas),index=b,dtype=float)

        for r in range(replicas):
        
            #print(r)
            print(f'{c}_{p}_{r}.csv.zip')
            file = f'{c}_{p}_{r}.csv.zip'
            M=pd.read_csv(os.path.join(TAB_DIR,file),index_col='taxa',compression='gzip')
            M=M/M.sum(axis=0)
            #M=np.log10( M.replace(0,np.nan) ).fillna(0)
            # occhio il replace influenza un botto se tieni i sample in comune!
            pearson_replica=np.triu(M.T.corr().values, k=1).flatten()
            #pearson_replica=pearson_replica[pearson_replica!=0]
            #pearson_replica=pearson_replica[pearson_replica!=1]
            h=np.histogram(pearson_replica,bins=bins)
            f=h[0]/(h[0].sum()*db)
            pk=f.copy()
            w=np.where(f>0)
            F_model = pd.Series(index=b,data=f)
            bx,f=b[w],f[w]

            P_avg[r]=pd.Series(index=bx,data=f)
            R[p].loc[c,r]= an.KLDiv(x=pearson_replica,y=pearson_data,b=bins)
            ax[i][j].plot(bx,f,color=col['model'][p],linewidth=1,alpha=0.4)
    

        ax[i][j].plot(b,P_avg.mean(axis=1),color=col['model'][p],linewidth=3,label='Model')

        ax[i,j].set_yscale('log')
        ax[i,j].set_xlim(-1,1)

        
ax[0,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))
ax[1,2].legend(fontsize=25,numpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.1))

OUTDIR=os.path.join(config['model']['folder'])

PLOT_DIR=os.path.join(OUTDIR,f'plots/')
es.check_path(PLOT_DIR)

Rtot= pd.DataFrame( columns = pd.MultiIndex.from_tuples([(p,r) for p in Pheno['phenotype'] for r in range(replicas)] ) )

TEST_DIR=os.path.join(OUTDIR,f'test/')

for p in Pheno['phenotype']: Rtot[p]=R[p]

Rtot.to_csv(os.path.join('../..',TEST_DIR,'R2_taxa_pearson.csv'),index=True)


fig.savefig(os.path.join('../..',PLOT_DIR,'pearson'), transparent=False, dpi=150,bbox_inches='tight')

print('done')