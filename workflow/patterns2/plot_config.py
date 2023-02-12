
def phenotype_legend(pheno):

    if pheno=='diagnosis':
        return {'H':'Healthy','U':'Disease'}

def phenotype_color(pheno):

    if pheno=='diagnosis':
        return {'data':
                    {'H':'#72A0C1','U':'#B81D27'},
                'model':
                    {'H':'#B284BE','U':'#4D7B41'} }

def ebar_style():

    return { 'ls':'none',
             'capsize':7,
             'capthick':2,
             'linewidth':3,
             'markeredgewidth':2,
             'marker':'o',
             'ms':15,
             'markeredgecolor':'black' }

def box_style():

    return {"boxstyle":"round",
            "fc":"#FFFFFF"}

def model_label(scaling_family):

    if scaling_family=='taylor':    return 'schlomilch'
    elif scaling_family=='poisson': return 'dirichlet'
    else: 
        print('Unknown scaling family')
        return ''