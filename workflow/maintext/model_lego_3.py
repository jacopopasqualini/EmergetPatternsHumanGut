import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os
import sys
import json
import pandas as pd
import plot_config

config_file = sys.argv[1]

with open(f'../../configurations/{config_file}.json') as json_file: 
    
    config = json.load(json_file)

data_config = config['data']

engine   = data_config['engine']
phen     = data_config['phenotype']
database = data_config['database']
protocol = data_config['protocol']
scale    = data_config['scale'] 
cuts     = [1e-06,1e-05,5e-05,1e-04]#data_config['cuts']

S={cuts[0]:'Low thr.$\\approx 10^{-6}$',
   cuts[2]:'High/High thr.$\\approx 10^{-4}$',
   cuts[3]:'High thr.$\\approx 10^{-4}$'}

patterns=['ao','occupancy','rarefaction']#,'pearson']
models=['schlomilch','dirichlet','cricket']

lab_pattern={'ao':'AO-relation','occupancy':'Occupancy','rarefaction':'Rarefaction'}#,'pearson':'Pearson Corr.'}
lab_model={'dirichlet':'$Dirichlet$','schlomilch':'$Schlomlich$','cricket':'$Grilli$'}
lab_pheno={'H':'$Healthy$','U':'$Unhealthy$'}

cuts=list(S.keys())

npatterns=len(patterns)
ncuts=len(cuts)
pheno=['H','U']
npheno = len(pheno)
nmodels=len(models)  
ylabels = list(lab_pattern.values())
xlabels = list(S.values())
             
style=True

fig, ax = plt.subplots(nrows=nmodels,ncols=npheno,figsize=(80,30))

fig.patch.set_facecolor('#FFFFFF')

x, y = np.meshgrid(np.arange(ncuts), np.arange(npatterns))
R = 0.4*np.ones(shape=(ncuts, npatterns))
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]

for m,i in zip(models,range(nmodels)):
    
    with open(f'../../configurations/{m}.json') as json_file: 
    
        model_config = json.load(json_file)

    OBS_DIR = os.path.join('../..',model_config["model"]["folder"],'test')

    for p,j in zip(pheno,range(npheno)):
        
        print(m,p)

        if True:

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

            if i<len(models)-1: 
                xticklabels=False
            else: 
                xticklabels=True

            print( ncuts, npatterns,xlabels,ylabels )
            ax[i,j].set(xticks=np.arange(ncuts), yticks=np.arange(npatterns), xticklabels=xlabels, yticklabels=ylabels[::-1])
            ax[i,j].set_xticks(np.arange(ncuts+1)-0.5, minor=True)
            ax[i,j].set_yticks(np.arange(npatterns+1)-0.5, minor=True)
            ax[i,j].grid(which='minor',color='grey')
            ax[i,j].set_aspect('equal', adjustable='box')

        L=pd.DataFrame(index=patterns,columns=cuts,dtype=float)
        V=pd.DataFrame(index=patterns,columns=cuts,dtype=float)
        for o in patterns:
        
            r2_file=os.path.join(OBS_DIR,f'R2_{o}.csv')
            r=pd.read_csv(r2_file,index_col=0,header=[0, 1])
            r=r.replace(np.inf,0)
            for c in cuts:
                
                q=r[p].loc[c]
                L.loc[o,c]=q[q!=np.inf].mean()
                V.loc[o,c]=r[p].loc[c].std()

        c = L.values

        u=c.copy()
        u=u[::-1,:]

        # ocean, twilight, cubehelix, gist_heat
        col = PatchCollection(circles, array=u.flatten(), cmap="gist_heat",edgecolors='grey',linewidth=1.5)
        
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

            if i<len(models)-1: 
                ax[i,j].set_xticklabels([])        
            else: 
                ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(),fontsize=25,rotation=30,fontstyle='italic')

            
            if j<1:
                ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(),fontsize=25,rotation=30,fontstyle='italic')
            else:
                ax[i,j].set_yticklabels([])        

plt.subplots_adjust(left=0,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=-0.6, 
                    hspace=0.1)

fig.savefig(os.path.join('../../plots/maintext','model_comparison_3'), transparent=False, dpi=150,bbox_inches='tight')
