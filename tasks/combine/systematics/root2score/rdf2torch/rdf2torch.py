from root2score.rdf2torch.ak_parser import parse, to_ak
from root2score.rdf2torch.torch_dataset import EventsDataset
from root2score.rdf2torch.additional_data import add_additional_data
import numpy as np


vars_region_dict={
    "Muons":{
        "Jet":["Jet_pt",
                "Jet_phi",
                "Jet_eta",
                "Jet_btagDeepFlavB",
                "Jet_btagDeepFlavCvB",
                "Jet_btagDeepFlavCvL"],
        "SecondLept":["SecondLept_pt",
                      "SecondLept_phi",
                      "SecondLept_eta"],
        "Lepton":["Muon_pt[0]",
                "Muon_phi[0]",
                "Muon_eta[0]"],
        "MET" : ["MET_pt",
                "MET_phi",
                "MET_eta"],
        "Masses":["Masses"]
        },
    "Electrons":{
        "Jet":["Jet_pt",
                "Jet_phi",
                "Jet_eta",
                "Jet_btagDeepFlavB",
                "Jet_btagDeepFlavCvB",
                "Jet_btagDeepFlavCvL"],
        "SecondLept":["SecondLept_pt",
                    "SecondLept_phi",
                    "SecondLept_eta"],
        "Lepton":["Electron_pt[0]",
                "Electron_phi[0]",
                "Electron_eta[0]"],
        "MET" : ["MET_pt",
                "MET_phi",
                "MET_eta"],
        "Masses":["Masses"]
        }
}

additional=["LeptLabel","HadDecay","AdditionalPartons","Weights"]





def rdf2torch(rdf,
              region=None,
              weight_syst_list=None,
              nEv_lumi=None,
              eff=None):


    var_dict=vars_region_dict[region]

    print("",flush=True)
    rdf[0].Report().Print()
    
    if weight_syst_list is None:
        ak_arrays=to_ak(rdf,var_dict)
    else:
        additional_columns=["Weights_"+weight_syst for weight_syst in weight_syst_list]
        ak_arrays=to_ak(rdf,var_dict,others=additional_columns)

    print("\nConverting to torch tensors...",flush=True)
    torch_dict=parse(ak_arrays,var_dict)
    dataset=EventsDataset()
    
    for key in torch_dict:
        if key!="Masses":
            dataset.add_data(key,torch_dict[key],var_dict[key])
        else:
            infos=["WLept"]+["TLept"]*7+["Jet"]+["WHad"]*6+["Jet"]+["WHad"]*5+["Jet"]+["WHad"]*4+["Jet"]+["WHad"]*3+["Jet"]+["WHad"]*2+["Jet"]+["WHad"]*1+["Jet"]
            dataset.add_data(key,torch_dict[key],infos)
            

    if weight_syst_list is None:
        weights=ak_arrays["Weights"].to_numpy()
        weights=weights*nEv_lumi*eff/len(weights)
    else:
        w_nom=ak_arrays["Weights"].to_numpy()
        weights={"nominal":w_nom*nEv_lumi*eff/len(w_nom)}
        for weight_syst in weight_syst_list:
            weights[weight_syst]=ak_arrays["Weights_"+weight_syst].to_numpy()*nEv_lumi*eff/len(w_nom)

    return dataset,weights
