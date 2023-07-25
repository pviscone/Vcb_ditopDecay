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

def print_log(weights,sum_of_preselection_weights):
    print(f"Total efficiency: {(np.sum(weights))*100:.2f}%\n",flush=True) # type: ignore
    print(f"len_events: {len(weights)}",flush=True)
    print(f"Sum weights: {(np.sum(weights)*sum_of_preselection_weights):.2f}",flush=True) # type: ignore
    print(f"Sum weights/len_events: {(np.sum(weights)*sum_of_preselection_weights/len(weights))*100:.2f}%\n",flush=True) # type: ignore




def rdf2torch(rdf,cut=None,generator=None, weight_syst_list=None,sum_of_preselection_weights=None,real_nevent=None):
    assert (cut=="Muons" or cut=="Electrons") #"cut must be Muons or Electrons"
    assert sum_of_preselection_weights is not None
    if cut=="Muons":
        var_dict=mu_vars
    else:
        var_dict=ele_var
    print("",flush=True)
    #print(f"- - - - - {cut} - - - - -",flush=True)
    rdf[0].Report().Print()
    if weight_syst_list is None:
        ak_arrays=to_ak(rdf,var_dict)
    else:
        additional_columns=["Weights_"+syst for syst in weight_syst_list]
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
            
    #dataset=add_additional_data(dataset,ak_arrays["LHEPart_pdgId"],additional_list=additional,generator=generator)
    #dataset.add_additional_info("generator",generator)
    if weight_syst_list is None:
        weights=ak_arrays["Weights"].to_numpy()/sum_of_preselection_weights
        print_log(weights,sum_of_preselection_weights)
        

    else:
        weights={"nominal":ak_arrays["Weights"].to_numpy()/sum_of_preselection_weights}
        print(f"\nN_MC/N_Run2: {(sum_of_preselection_weights/real_nevent):.2f}",flush=True)
        print_log(weights["nominal"],sum_of_preselection_weights)
        for syst in weight_syst_list:
            weights[syst]=ak_arrays[f"Weights_{syst}"].to_numpy()/sum_of_preselection_weights
            print(f"{syst} efficiency: {(np.sum(weights[syst])*100):.2f}%",flush=True) # type: ignore

    
    return dataset,weights
