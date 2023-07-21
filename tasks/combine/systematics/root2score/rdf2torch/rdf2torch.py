from root2score.rdf2torch.ak_parser import parse, to_ak
from root2score.rdf2torch.torch_dataset import EventsDataset
from root2score.rdf2torch.additional_data import add_additional_data
import numpy as np
mu_vars={"Jet":["Jet_pt",
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
        }



ele_var={"Jet":["Jet_pt",
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

additional=["LeptLabel","HadDecay","AdditionalPartons"]





def rdf2torch(rdf,cut=None,generator=None):
    assert (cut=="Muons" or cut=="Electrons") #"cut must be Muons or Electrons"
    if cut=="Muons":
        var_dict=mu_vars
    else:
        var_dict=ele_var
    print("",flush=True)
    #print(f"- - - - - {cut} - - - - -",flush=True)
    rdf[0].Report().Print()
    ak_arrays=to_ak(rdf,var_dict)
    print("\nConverting to torch tensors...",flush=True)
    torch_dict=parse(ak_arrays,var_dict)
    dataset=EventsDataset()
    
    for key in torch_dict:
        if key!="Masses":
            dataset.add_data(key,torch_dict[key],var_dict[key])
        else:
            infos=["WLept"]+["TLept"]*7+["Jet"]+["WHad"]*6+["Jet"]+["WHad"]*5+["Jet"]+["WHad"]*4+["Jet"]+["WHad"]*3+["Jet"]+["WHad"]*2+["Jet"]+["WHad"]*1+["Jet"]
            dataset.add_data(key,torch_dict[key],infos)
            
    #dataset=add_additional_data(dataset,ak_arrays["LHEPart_pdgId"],additional_list=additional,generator=generator)
    #dataset.add_additional_info("generator",generator)
    
    weights=ak_arrays["Weights"].to_numpy()
    weights=weights/np.sum(weights)
    print(f"Total number of selected events: {len(weights)}",flush=True)
    return dataset,weights
  
    
