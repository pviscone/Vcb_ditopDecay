#%%
import json


#paths=['Combined']
#modifier2=0.965

#paths=['Muons']
#modifier2=1.01

paths=['Electrons']
modifier2=1.04

for path in paths:
    print(f'########### {path} ###########')
    jdict = json.load(open(path+'/datacard/impacts.json'))['params']
    poi=json.load(open(path+'/datacard/impacts.json'))['POIs'][0]
    up=(poi['fit'][1]-poi['fit'][0])
    down=(poi['fit'][2]-poi['fit'][1])
    mean=(up+down)/2
    res={}
    for syst in jdict:
        res[syst['groups'][0]]=[]
        
    for syst in jdict:
        temp={'name':syst['name'],'impact_r':syst['impact_r']}
        res[syst['groups'][0]].append(temp)

    tot=0
    for group in res:
        impact=0
        for source in res[group]:
            impact+=modifier2*source['impact_r']**2
            tot+=modifier2*source['impact_r']**2
        print(group,impact**0.5)
    print('Syst',tot**0.5)
    print('TotalDown',down)
    print('TotalUp',up)
    print('TotalMean',up)
    print('statDown',(down**2-tot)**0.5)
    print('statMean',(mean**2-tot)**0.5)
    
#%%
import numpy as np
def f(*args):
    return np.sqrt(np.sum(np.array(args)**2))
    