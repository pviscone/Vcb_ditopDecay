import numpy as np
import torch
import awkward as ak
import ROOT

ROOT.EnableImplicitMT()




def dict2list(dictionary):
    l=[]
    for key in dictionary:
        for column in dictionary[key]:
            l.append(column.split("[")[0])
    return l


def to_ak(rdf_list,var_dict,others=None):
    columns=dict2list(var_dict)+["LHEPart_pdgId","Weights"]
    if others is not None:
        columns+=others
    ak_arrays=ak.from_rdataframe(rdf_list[0],columns)
    for rdf in rdf_list[1:]:
        ak_arrays=ak.concatenate([ak_arrays,ak.from_rdataframe(rdf,columns)])
    return ak_arrays
    
    
def parse(ak_arrays,var_dict):
    res={}
    
    for key in (var_dict.keys()):
        features=var_dict[key]
        for idx, feature in enumerate(features):
            feat_split=feature.split("[")
            feature=feat_split[0]

            ak_array=ak_arrays[feature]

            if len(feat_split)>1:
                index=int(feat_split[1].split("]")[0])
                new_column=torch.tensor(ak_array[:,index,None].to_numpy()[:,:,None])
            else:
                shape0=len(ak_array)
                if ak_array.ndim==1:
                    new_column=ak_array.to_numpy()[:,None]
                    shape1=1
                else:
                    new_column=np.array(ak_array.layout.content)
                    shape1=ak_array.layout.offsets[1]

                new_column=torch.tensor(new_column)
                new_column=torch.reshape(new_column,(shape0,shape1,1))

            if idx==0:
                res[key]=new_column
            else:
                dim=2
                res[key]=torch.cat((res[key],new_column),dim=dim)
    return res
