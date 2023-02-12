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

OBS_DIR = os.path.join('../..',config['model']['folder'],'observables')
MOD_DIR = plot_config.model_label(config['model']['opts']['scaling_family'])
print('>>>>>>>>>>>>>',MOD_DIR)
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

title=['Low thr.$\\approx 10^{-6}$','Mid thr.$\\approx 10^{-5}$','Mid/high thr.$\\approx 10^{-5}$','High thr.$\\approx 10^{-4}$']

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
    
            ax[1,j].set_title(title[j],fontsize=25)


        md=Pheno['X'][p][c]['original mean']['original']
        od=Pheno['X'][p][c]['binary mean']['original']
        md,od=np.log10(md),od
        
        AO_data=pd.DataFrame()
        AO_data['x'],AO_data['y']=md,od
        AO_data=AO_data.dropna()
        
        #ax[i][j].scatter(AO_data['x'],AO_data['y'],alpha=0.01,s=80,color=col['data'][p],zorder=1,edgecolor='none')
        P=pl.binning(x=AO_data['x'],y=AO_data['y'],n_bins=12)
        ax[i][j].errorbar(x=P['x_mean'],xerr=P['x_std'],y=P['y_mean'],yerr=P['y_std'],color=col['data'][p],**ebar,zorder=3,label=lab[p])
    
        M_model,O_model=np.zeros(1),np.zeros(1)
        
        for r in range(replicas):

            print(r)
            file = f'{c}_{p}_{r}.csv.zip'
            MX=pd.read_csv(os.path.join(OBS_DIR,file),index_col=0,header=[0, 1],compression='gzip')
            
            mm=np.log10( MX['original mean']['original'] )
            om=MX['binary mean']['original']
            #ax[i][j].scatter(mm,om,alpha=0.01,s=80,color=col['model'][p],zorder=0,edgecolor='none')
            M_model,O_model=np.append(M_model,mm),np.append(O_model,om)
            AO_model=pd.DataFrame()
            AO_model['x'],AO_model['y']=mm,om
            AO_model=AO_model.dropna()
            #AO_model=AO_model[AO_model['x']<0]

            R2.loc[c,r]=an.compare_neutral_curves(AO_data,AO_model,delta=0.2,n_bins=20)


        P=pl.binning(x=AO_model['x'],y=AO_model['y'],n_bins=40)
        
        #yl=np.array(P['y_mean'])-np.array(P['y_std'])
        #yu=np.array(P['y_mean'])+np.array(P['y_std'])
        
        #ax[i][j].fill_between(x=np.array(P['x_mean']),y1=yl,y2=yu,color=col['model'][p],alpha=0.5,zorder=1)
        ax[i][j].plot(P['x_mean'],P['y_mean'],color=pl.complementary_rgb(col['model'][p]),zorder=2,linewidth=5.5,label='Model')
        
        r2=round(R2.loc[c].mean(),3)
        ax[i,j].text(fontsize=25,s=f'$R^2$={r2}',x=-2.725,y=0.65,bbox=box )
        
        ax[i][j].set_xlim(-4,6.5)
        ax[i][j].set_ylim(-0.2,1.2)
        
        if j==2:
            ax[i,j].legend(fontsize=22.5,numpoints=1 ,fancybox=True,shadow=True,loc=(0.1,0.7))

        
    R[p]=R2

ax[1,1].set_xlabel('$Mean$ $Abundance$, $\log_{10} \overline{\\mu}$',fontsize=45)
ax[1,0].text(s='$Occupancy$, $\overline{o}$',rotation=90,x=-8,y=0.80,fontsize=45)

OUTDIR=os.path.join(config['model']['folder'])

PLOT_DIR=os.path.join(OUTDIR,f'plots/')
es.check_path(PLOT_DIR)

fig.savefig(os.path.join('../..',PLOT_DIR,'ao'), transparent=False, dpi=150,bbox_inches='tight')

Rtot= pd.DataFrame( columns = pd.MultiIndex.from_tuples([(p,r) for p in Pheno['phenotype'] for r in range(replicas)] ) )

for p in Pheno['phenotype']: Rtot[p]=R[p]

TEST_DIR=os.path.join(OUTDIR,f'test/')
es.check_path(TEST_DIR)

Rtot.to_csv(os.path.join('../..',TEST_DIR,'R2_ao.csv'),index=True)

print('done')