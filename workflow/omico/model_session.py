import table as tb
#import plot as pl
import sys

sys.path.append('../model')

import ecomodel as em 

import pandas as pd
import numpy as np
import os
import subprocess
import json

PROJ_ROOT='../..'

colorX={('all','kaiju','RefSeq','relative_abundance'):'#587065',
        ('H','kaiju','RefSeq','corePFAM'):'#FF5500',
        ('U','kaiju','RefSeq','corePFAM'):'#5D8AA8',
        ('H','kaiju','RefSeq','relative_abundance'):'#D6C7B8',
        ('U','kaiju','RefSeq','relative_abundance'):'#BF2443',
        ('all','kraken','RefSeq','relative_abundance'):'#6A7310',
        ('H','kraken','RefSeq','relative_abundance'):'#FFD500',
        ('U','kraken','RefSeq','relative_abundance'):'#59B300'}

def check_path( target_path ):

    if os.path.exists(target_path):  pass
    
    else: 

        f=PROJ_ROOT

        for s in target_path.split('/'):

            f=os.path.join(f,s)
            if os.path.exists(f):
                print(f'{f}: folder exists')
            else: 
                print(f'mkdir {f}')
                subprocess.run(['mkdir',f])

    out_path = os.path.join(PROJ_ROOT,target_path)

    if os.path.exists(out_path): 
        print(f'<<{out_path}>>: available')
        return out_path
    else: 
        print(f'<<{out_path}>> creation: failed')


def model_session(data_session,cuts,replicas,DIR,configuration,model,specifics={},mad=None,taylor=None):
    
    MS = {p:data_session[p] for p in ['engine', 'database', 'phenotype', 'partition','protocol','group']}
    
    engine=MS['engine']
    database=MS['database']
    phenotype=MS['phenotype']
    protocol=MS['protocol']
    print(f'SESSION > engine:{engine} / databse:{database} / group:{phenotype} / protocol:{protocol}\n')

    SAMPLES_DIR = check_path( os.path.join(DIR,f'samples') )
    OBSERVABLES_DIR =  check_path( os.path.join(DIR,f'observables') )

    tables,observables = {},{}

    phenotype_parameters={} 

    for p in MS['phenotype']:

        phenotype_parameters[p]={}

        print('\n'+f'Phenotype: {p}')
        
        # get data without the cutoff
        print(cuts,data_session['T'][p].keys())
        T0 = data_session['T'][p][ cuts[0] ]
        
        # initialize model object
        if model!='cricket': 
            Gp = em.CompoundSchlomilch(D=T0)
        else:                
            Gp = em.Cricket(D=T0)
        
        # set taylor family taylor
        if specifics['scaling_family']=='taylor': 
            tau=2 
        elif specifics['scaling_family']=='poisson': 
            tau=1

        Gp.fit_taylor(tau=tau)
        Gp.fit_mad(scale=specifics['mad_scale'],ensemble=200,cut=-50,model=specifics['mad_model'],cut_field='loc')
       
        tp,xp = {},{}
        
        for c in cuts:
            
            if specifics['mad']=='fit':
                Gp.sample_parameters(mode='random',samples=100000)
            elif specifics['mad']=='empirical':
                Gp.sample_parameters(mode='empirical')
            elif specifics['mad']=='custom':
                Gp.sample_parameters(mode='write',samples=mad)

            Experiments = pd.DataFrame( index= Gp.data.components )
            Experiments.index=Experiments.index.rename(Gp.data.annotation)

            xp[c]={}
            
            for r in range(replicas):
            
                print(f'replica:{r} > ',end='')

                Gp.sample_model(rank_conservation=True,ra_cut=c )
                
                E = Gp.sample.form['original']
                E.index=E.index.rename(Gp.data.annotation)

                file = f'{c}_{p}_{r}.csv.zip'
                E.to_csv(os.path.join(SAMPLES_DIR,file),compression="gzip")
                Gp.sample_observables.to_csv(os.path.join(OBSERVABLES_DIR,file),compression="gzip")
                
            print()
            
            Experiments=Experiments.fillna(0)
                
            tp[c]=tb.table(Experiments)
                
        tables[p]=tp
        observables[p]=xp

        phenotype_parameters[p]['taylor']=Gp.taylor
        phenotype_parameters[p]['mad']=Gp.mad
    
    configuration['model_parameters']=phenotype_parameters

    json_object = json.dumps(configuration)
 
    with open(os.path.join('../..',DIR,"config.json"), "w") as outfile:
        outfile.write(json_object)

    R = {'engine':engine,
         'database':database,
         'phenotype':MS['phenotype'],
         'partition':MS['partition'],
         'protocol':MS['protocol'],
         'T':tables,
         'X':observables,
         'color':'#B284BE' }
               
    return R  

'''
def bayesian_session(data_session,model,scale,cuts,replicas,DIR,configuration,specifics={},mad=None,taylor=None):
    
    MS = {p:data_session[p] for p in ['engine', 'database', 'phenotype', 'partition','protocol','group']}
    
    engine=MS['engine']
    database=MS['database']
    phenotype=MS['phenotype']
    group=MS['group']
    protocol=MS['protocol']
    print(f'SESSION > engine:{engine} / databse:{database} / group:{phenotype} / protocol:{protocol}\n')

    BAYES_DIR = os.path.join('../../results',model,scale)
    bayes = pd.read_csv(os.path.join(BAYES_DIR,'{model}_summary.tsv'),sep='\t')
    SAMPLES_DIR = check_path( os.path.join(DIR,f'samples') )
    OBSERVABLES_DIR =  check_path( os.path.join(DIR,f'observables') )

    tables,observables = {},{}

    phenotype_parameters={} 

    for p in MS['phenotype']:

        phenotype_parameters[p]={}

        print('\n'+f'Phenotype: {p}')
        
        # get data without the cutoff
        print(cuts,data_session['T'][p].keys())
        T0 = data_session['T'][p][ cuts[0] ]
        X0 = data_session['X'][p][ cuts[0] ]
        
        # initialize model object
        Gp = em.CompoundSchlomilch(D=T0)
        
        # fit taylor
        if specifics['scaling_family']=='taylor': tau=2 
        elif specifics['scaling_family']=='poisson': tau=1
        else: fit_taylor=True

        scale=specifics['mad_scale']
        xt=(X0[f'{scale} mean']['original'].values)
        xt=np.log(xt[xt>0])
        yt=(X0[f'{scale} var']['original'].values)
        yt=np.log(yt[yt>0])

        if specifics['taylor']=='empirical':
            Gp.fit_taylor(fit=False,taylor={'slope':tau,'intercept':np.exp(yt.mean()-tau*xt.mean())})
        elif specifics['taylor']=='fit': 
            Gp.fit_taylor(fit=fit_taylor)
        elif specifics['taylor']=='custom':
            Gp.fit_taylor(fit=False,taylor=specifics['taylor'])
        
        if specifics['mad']=='fit':
            Gp.fit_mad(scale=specifics['mad_scale'],ensemble=200,cut=-50,model=specifics['mad_model'],cut_field='loc')
        elif specifics['mad']=='empirical':
            Gp.fit_mad(fit=False) 
        elif specifics['mad']== 'custom':
            Gp.fit_mad(fit=False,mad=specifics['mad_model'])    

        tp,xp = {},{}
        
        for c in cuts:
            
            if specifics['mad']=='fit':
                Gp.sample_parameters(mode='random',samples=100000)
            elif specifics['mad']=='empirical':
                Gp.sample_parameters(mode='empirical')
            elif specifics['mad']=='custom':
                Gp.sample_parameters(mode='write',samples=mad)

            Experiments = pd.DataFrame( index= Gp.data.components )
            Experiments.index=Experiments.index.rename(Gp.data.annotation)

            xp[c]={}
            
            for r in range(replicas):
            
                print(f'replica:{r} > ',end='')
                if MS['protocol']=='corePFAM':

                    U=data_session['T'][p].form['relative'].replace(0,np.nan).values.flatten()
                    ra_c = U[~np.isnan(U)].min()

                else: ra_c=c

                Gp.sample_model(rank_conservation=True,ra_cut=ra_c )
                
                E = Gp.sample.form['original']
                E.index=E.index.rename(Gp.data.annotation)

                file = f'{c}_{p}_{r}.csv.zip'
                E.to_csv(os.path.join(SAMPLES_DIR,file),compression="gzip")
                Gp.sample_observables.to_csv(os.path.join(OBSERVABLES_DIR,file),compression="gzip")
                
            print()
            
            Experiments=Experiments.fillna(0)
                
            tp[c]=tb.table(Experiments)
                
        tables[p]=tp
        observables[p]=xp

        phenotype_parameters[p]['taylor']=Gp.taylor
        phenotype_parameters[p]['mad']=Gp.mad
    
    configuration['model_parameters']=phenotype_parameters

    json_object = json.dumps(configuration)
 
    with open(os.path.join('../..',DIR,"config.json"), "w") as outfile:
        outfile.write(json_object)

    R = {'engine':engine,
         'database':database,
         'phenotype':MS['phenotype'],
         'partition':MS['partition'],
         'protocol':MS['protocol'],
         'T':tables,
         'X':observables,
         'color':'#B284BE' }
               
    return R  
'''