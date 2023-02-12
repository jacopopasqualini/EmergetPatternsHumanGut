import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os
import sys
import json
import pandas as pd
import plot_config

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

OBS_DIR = '../../data/observables'
MOD_DIR = plot_config.model_label(config['model']['opts']['scaling_family'])

S={cuts[0]:'Low thr.$\\approx 10^{-6}$',
   cuts[1]:'Mid thr.$\\approx 10^{-5}$',
   cuts[2]:'High thr.$\\approx 10^{-4}$'}

patterns=['ao','occupancy','rarefaction']
models=['schlomilch','dirichlet']

lab_pattern={'ao':'AO-relation','occupancy':'Occupancy','rarefaction':'Rarefaction'}
lab_model={'dirichlet':'$Dirichlet$','schlomilch':'$Schlomlich$'}
lab_pheno={'H':'$Healthy$','U':'$Unhealthy$'}

cuts=list(S.keys())

npatterns=3
ncuts=len(cuts)
pheno=['H','U']
npheno = len(pheno)
nmodels=len(models)  
ylabels = list(lab_pattern.values())
xlabels = list(S.values())
             
replicas=20
style=True

fig, ax = plt.subplots(nrows=nmodels,ncols=npheno,figsize=(15,15))


fig.patch.set_facecolor('#FFFFFF')

x, y = np.meshgrid(np.arange(ncuts), np.arange(npatterns))
R = 0.4*np.ones(shape=(ncuts, npatterns))#s/s.max()/2
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]

for m,i in zip(models,range(nmodels)):
    
    for p,j in zip(pheno,range(npheno)):
        
        print(m,p)

        if True:
            
            #if i==1: xl=xlabels
            #else: xl=[]

            #if j==0: yl=ylabels[::-1]
            #else: yl=[]

            for axis in ['bottom','left','top','right']:  ax[i,j].spines[axis].set_linewidth(2.5)   
            ax[i,j].tick_params(left=False, bottom=False,right=False,top=False)

            ax[i,j].tick_params(right=False,top=False,bottom=False,left=False)

            if i==0:
                ax[i,j].xaxis.set_label_position("top")
                ax[i,j].set_xlabel(lab_pheno[p],fontsize=45,labelpad=15)

            if j==1:
                ax[i,j].yaxis.set_label_position("right")
                ax[i,j].set_ylabel(lab_model[m],rotation=270,labelpad=50,fontsize=45)

            if j>0: 
                yticklabels=False
            else: 
                yticklabels=True

            
            ax[i,j].set(xticks=np.arange(ncuts), yticks=np.arange(npatterns), xticklabels=xlabels, yticklabels=ylabels[::-1])
            ax[i,j].set_xticks(np.arange(ncuts+1)-0.5, minor=True)
            ax[i,j].set_yticks(np.arange(npatterns+1)-0.5, minor=True)
            ax[i,j].grid(which='minor')
            ax[i,j].set_aspect('equal', adjustable='box')

        L=pd.DataFrame(index=patterns,columns=cuts,dtype=float)
        V=pd.DataFrame(index=patterns,columns=cuts,dtype=float)
        
        for o in patterns:
        
            r2_file=os.path.join(OBS_DIR,o,f'R2_{phen}_{engine}_{database}_{m}.csv')
            r=pd.read_csv(r2_file,index_col=0,header=[0, 1])
            
            for c in cuts:
                
                L.loc[o,c]=r[p].loc[c].mean()
                V.loc[o,c]=r[p].loc[c].std()

        c = L.values

        u=c.copy()
        u=u[::-1,:]

        col = PatchCollection(circles, array=u.flatten(), cmap="Oranges_r",edgecolors='black',linewidth=1)
        
        print(col.get_facecolor())
        #print(K.shape)

        col.set_clim([-1, 1])

        
        ax[i,j].add_collection(col)
        
        z=c.T

        for k in range(ncuts):

            for l in range(npatterns):
                
                g=round(z[k,-l-1],3)
                
                text_color='black'
                ax[i,j].text(x=k-0.25,y=l-0.05,s=f'{g}',fontsize=25,fontstyle='oblique',color=text_color)

        
        if True:

            if i<1: 
                ax[i,j].set_xticklabels([])        
            else: 
                ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(),fontsize=25,rotation=30,fontstyle='italic')

            
            if j<1:
                ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(),fontsize=25,rotation=30,fontstyle='italic')
            else:
                ax[i,j].set_yticklabels([])        

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.1)

fig.savefig(os.path.join('../../plots/maintext','model_comparison'), transparent=False, dpi=150,bbox_inches='tight')
