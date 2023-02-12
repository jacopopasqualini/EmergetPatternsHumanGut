import sys
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../model')
import ecomodel as em
  
sys.path.append('../omico')
import plot as pl
import fit as ft
import analysis as an
import table as tb
import session as es
import model_session as ms

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

    print(engine,phen,database,protocol,scale,cuts)

phen_cuts=[-np.inf,1e-6]
nbins,digits=0,0
phenotype_map=pd.Series(dtype=object)
re_group = {'H':'H','UC':'U','CD':'U','IBS-C':'U','IBS-D':'U'}

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

model_config=config['model']

SCH_DIR=model_config['folder']
model_opts=model_config['opts']
replicas = model_config['replicas']

config['original_file']=f'../../data/configurations/{config_file}'
config['date_time']=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

fig,ax=plt.subplots(ncols=2,figsize=(25,15))
fig.patch.set_facecolor('#FFFFFF')

for p,j in zip(Pheno['phenotype'],range(2)):

    print('\n'+f'Phenotype: {p}')
    
    T0 = Pheno['T'][p][ cuts[0] ]

    DataSize = T0.form['original'].sum(axis=0)
    DataDivs = T0.form['binary'].sum(axis=0)
    ax[j].scatter(DataSize,DataDivs,color='red',label='data')
    print(T0.observables)
    #DataMAD=np.log10(T0.observables['original']['relative mean']['original'])
    #DataVAD=np.log10(T0.observables['original']['relative var']['original'])
    #ax[j].scatter(DataMAD,DataVAD,color='red')
    #ax[j].hist(DataMAD,bins=15,histtype='step',color='red',linewidth=2)

    
    model = 'schlomilch'

    #fit the model with the data
    if True:
        if model=='schlomilch' or model=='cricket':
            tau=2
        elif model=='dirichlet':
            tau=1
        # initialize model object
        if model!='cricket': 
            Gp = em.CompoundSchlomilch(D=T0)
        else:                
            Gp = em.Cricket(D=T0)

        Gp.fit_taylor(tau=tau,report=True)
        Gp.fit_mad(scale='relative',report=True,ensemble=200,cut=-50,model='log-laplace',cut_field='loc')
        Gp.sample_parameters(mode='random',samples=100000)
        Gp.sample_model(rank_conservation=True,ra_cut=-np.inf )

        SampleSize = Gp.sample.form['original'].sum(axis=0)
        SampleDivs = Gp.sample.form['binary'].sum(axis=0)
        ax[j].scatter(SampleSize,SampleDivs,color='blue')

        #SamlpeMAD=np.log10(Gp.sample_observables['relative mean']['original'])
        #SampleVAD=np.log10(Gp.sample_observables['relative var']['original'])
        #ax[j].scatter(SamlpeMAD,SampleVAD,color='blue')
        #ax[j].hist(SamlpeMAD,bins=15,histtype='step',color='blue',linewidth=2)

    # generate model with just the same sample size
    if True:
        Dummy = pd.DataFrame(index=T0.components,columns=T0.samples)
        f=np.random.uniform(0,1,Dummy.shape[0])
        f=f/f.sum()
        N=T0.form['original'].sum(axis=0)

        for n in T0.samples:

            Dummy[n]=np.random.multinomial(n=int(N[n]),pvals=f)

        Dummy.index=Dummy.index.rename('taxa')

        if model=='schlomilch' or model=='cricket':
                tau=2
        elif model=='dirichlet':
            tau=1
        # initialize model object
        if model!='cricket': 
            Gp_check = em.CompoundSchlomilch(D=tb.table(Dummy))
        else:                
            Gp_check = em.Cricket(D=tb.table(Dummy))

        Gp_check.fit_taylor(taylor=Gp.taylor,write=True)
        Gp_check.fit_mad(mad=Gp.mad,write=True)
        Gp_check.sample_parameters(mode='random',samples=100000)

        Gp_check.sample_model(rank_conservation=True,ra_cut=cuts[0] )

        SampleSize = Gp_check.sample.form['original'].sum(axis=0)
        SampleDivs = Gp_check.sample.form['binary'].sum(axis=0)
        ax[j].scatter(SampleSize,SampleDivs,color='yellow')

        #SampleMAD=np.log10(Gp_check.sample_observables['relative mean']['original'])
        #SampleVAD=np.log10(Gp_check.sample_observables['relative var']['original'])
        #ax[j].scatter(SampleMAD,SampleVAD,color='yellow')
        #ax[j].hist(SampleMAD,bins=15,histtype='step',color='yellow',linewidth=2)


    # set the new model with dummy data
    if True:
        Dummy = pd.DataFrame(index=T0.components,columns=range(500))
        f=np.random.uniform(0,1,Dummy.shape[0])
        f=f/f.sum()
        for n,i in zip(np.logspace(4,9,500),range(500)):

            Dummy[i]=np.random.multinomial(n=n,pvals=f)

        Dummy.index=Dummy.index.rename('taxa')

        if model=='schlomilch' or model=='cricket':
                tau=2
        elif model=='dirichlet':
            tau=1
        # initialize model object
        if model!='cricket': 
            Gp_check = em.CompoundSchlomilch(D=tb.table(Dummy))
        else:                
            Gp_check = em.Cricket(D=tb.table(Dummy))

        Gp_check.fit_taylor(taylor=Gp.taylor,write=True)
        Gp_check.fit_mad(mad=Gp.mad,write=True)
        Gp_check.sample_parameters(mode='random',samples=100000)
        Gp_check.sample_model(rank_conservation=True,ra_cut=-np.inf )

        #SampleMAD=np.log10(Gp_check.sample_observables['relative mean']['original'])
        #SampleVAD=np.log10(Gp_check.sample_observables['relative var']['original'])
        #ax[j].scatter(SampleMAD,SampleVAD,color='black')
        #ax[j].hist(SampleMAD,bins=15,histtype='step',color='black',linewidth=2)
        SampleSize = Gp_check.sample.form['original'].sum(axis=0)
        SampleDivs = Gp_check.sample.form['binary'].sum(axis=0)
        ax[j].plot(SampleSize,SampleDivs,color='black')

    ax[j].set_xscale('log')
   # ax[j].set_ylim(0,6e3)

fig.savefig('sar_check', transparent=False, dpi=150,bbox_inches='tight')



        