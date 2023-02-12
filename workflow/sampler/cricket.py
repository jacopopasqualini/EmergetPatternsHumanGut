import sys
import json
import pandas as pd
from datetime import datetime

sys.path.append('../model')
import ecomodel as em
  
sys.path.append('../omico')
import plot as pl
import fit as ft
import analysis as an
import table as tb
import model_session as ms
import session as es

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

phen_cuts=cuts
nbins,digits=0,0
phenotype_map=pd.Series(dtype=object)
re_group={}

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

print(config)
model_config=config['model']

CRK_DIR=model_config['folder']
model_opts=model_config['opts']
replicas = model_config['replicas']

config['original_file']=f'../../data/configurations/{config_file}'
config['date_time']=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

Model = ms.model_session(data_session=Pheno,cuts=cuts,specifics=model_opts,replicas=replicas,DIR=CRK_DIR,configuration=config,model=model_config['name'])