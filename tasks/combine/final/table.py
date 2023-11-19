#%%
import pandas as pd
import numpy as np
procs=['mainTTSL','TTSLTau','TTDL','tW','tq','WWJets','WJets']


regions=['Muons','Electrons','Combined']


def read_unc(path):

    f=open(path,'r').read()
    unc=float(f.split("'hi': ")[1].split(',')[0])-1
    return unc

df=pd.DataFrame(columns=['name']+regions)



for i in range(len(procs)):
    df_dict={'name':procs[i]}
    for region in regions:
        unc=read_unc(f"{region}/datacard{i+1}/plot1DScan.out")*100
        df_dict[region]=unc
    df.loc[len(df)]=df_dict

df['test']=1/np.sqrt(1/df['Muons']**2+1/df['Electrons']**2)
df['diff']=df['test']-df['Combined']