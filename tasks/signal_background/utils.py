import numpy as np
import awkward as ak
import sys
from coffea.nanoevents.methods import vector

sys.path.append("../../utils/coffea_utils")
from coffea_utils import Muon_cuts, np_and,np_or,MET_eta



def padded_matrix(ak_array, pad):
    masked_array = ak.pad_none(ak_array, pad, clip=True).to_numpy()
    masked_array.data[masked_array.mask] = 0
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

def select_muon_events(events,num_jet_to_select):
    muon_selector=Muon_cuts()
    events=muon_selector.process(events,out="events")
    
    events["Muon"] = events.Muon[:,0]
    events["MET"]=ak.zip(
        {
            "pt": events.MET.pt,
            "eta": MET_eta(events.Muon,events.MET),
            "phi": events.MET.phi,
            "mass": np.zeros_like(events.MET.pt),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    events["Jet"]=events.Jet[:,:num_jet_to_select]
    events["MET","WMass"]=(events.MET+events.Muon).mass
    events["Jet","TMass"]=(events.Jet+events.Muon+events.MET).mass
    return events