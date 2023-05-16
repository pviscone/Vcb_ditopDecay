import numpy as np
import awkward as ak



def build(coffea_obj,label=None,LHELept=None,num_jet_to_select=None):
    assert label is not None
    assert num_jet_to_select is not None
    if LHELept is not None:
        Lepton_mask=ak.sum(np.abs(coffea_obj.LHEPart.pdgId)==LHELept,axis=1)>0
        coffea_obj=coffea_obj[Lepton_mask]
    Muon_label=ak.sum(np.abs(coffea_obj.LHEPart.pdgId)==13,axis=1)>0
    Electron_label=ak.sum(np.abs(coffea_obj.LHEPart.pdgId)==11,axis=1)>0
    Tau_label=ak.sum(np.abs(coffea_obj.LHEPart.pdgId)==15,axis=1)>0
    
    Lept_label=np.zeros_like(Muon_label.to_numpy(),dtype=int)
    Lept_label[Muon_label.to_numpy()]=13
    Lept_label[Electron_label.to_numpy()]=11
    Lept_label[Tau_label.to_numpy()]=15
    
    Mu_feature=["pt", "eta", "phi"]
    Nu_feature=["pt", "eta", "phi","WLeptMass"]
    Jet_feature=["pt", "eta", "phi", "btagDeepFlavB",
                "btagDeepFlavCvB", "btagDeepFlavCvL", "TLeptMass","THadMass","WHadMass"]
    muon_matrix, mu_labels = build_matrix(coffea_obj,"Electron", Mu_feature,index=0)
    nu_matrix, nu_labels = build_matrix(coffea_obj,"MET", Nu_feature)
    jet_matrix, jet_labels = pad_and_alternate(coffea_obj,"Jet",
                                                        Jet_feature,
                                                        num_jet_to_select)
    matrix=np.hstack([
        muon_matrix,
        nu_matrix,
        jet_matrix,
        label*np.ones((muon_matrix.shape[0],1)),
        np.atleast_2d(Lept_label).T])
    col_labels=mu_labels+nu_labels+jet_labels+["label"]+["Lept_label"]
    return matrix,col_labels

def padded_matrix(ak_array, pad):
    masked_array = ak.pad_none(ak_array, pad, clip=True).to_numpy()
    masked_array.data[masked_array.mask] = -10
    return masked_array.data


def column(ak_array):
    return np.atleast_2d(ak_array.to_numpy(allow_missing=False)).T


def build_matrix(events,obj, variable_list,index=None):
    column_list = []
    col_labels = []
    for var in variable_list:
        col_labels.append(f"{obj}_{var}")
        if index is not None:
            var+=f"[:,{index}]"
        exec(f"column_list.append(column(events.{obj}.{var}))")
    return np.hstack(column_list), col_labels


def alternate_column(matrix_list):
    num_particles = matrix_list[0].shape[1]
    num_features = len(matrix_list)
    final_matrix = np.empty(
        (matrix_list[0].shape[0], num_particles*num_features))
    for feature in range(num_features):
        for obj in range(num_particles):
            final_matrix[:, num_features*obj +
                         feature] = matrix_list[feature][:, obj]
    return final_matrix


def pad_and_alternate(events,obj, variable_list, pad):
    matrix_list = []
    col_labels = []
    for var in variable_list:
        if var == "Tmass":
            matrix_list.append(padded_matrix(events.Jet.Tmass, pad))
        else:
            exec(
                f"matrix_list.append(padded_matrix(events.{obj}.{var},{pad}))")
    for i in range(pad):
        for var in variable_list:
            col_labels.append(f"{obj}{i}_{var}")
    return alternate_column(matrix_list), col_labels

