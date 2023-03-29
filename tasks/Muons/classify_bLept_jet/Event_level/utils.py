import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from matplotlib.patches import Circle


def padded_matrix(ak_array, pad):
    masked_array = ak.pad_none(ak_array, pad, clip=True).to_numpy()
    masked_array.data[masked_array.mask] = 0
    return masked_array.data


def column(ak_array):
    return np.atleast_2d(ak_array.to_numpy(allow_missing=False)).T


def build_matrix(events,obj, variable_list):
    column_list = []
    col_labels = []
    for var in variable_list:
        col_labels.append(f"{obj}_{var}")
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
            matrix_list.append(padded_matrix(events.Tmass, pad))
        else:
            exec(
                f"matrix_list.append(padded_matrix(events.{obj}.{var},{pad}))")
    for i in range(pad):
        for var in variable_list:
            col_labels.append(f"{obj}{i}_{var}")
    return alternate_column(matrix_list), col_labels


def np_and(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.bitwise_and(arg, res)
    return res


def circle(ax, x, y, color, label=None, fill=False, alpha=0.4, radius=0.4):
    ax.add_patch(Circle((x, y), radius=radius, color=color,
                 label=label, alpha=alpha, fill=fill))
    ax.add_patch(Circle((x, y-6.28), radius=radius, color=color,
                 label=None, alpha=alpha, fill=fill))
    ax.add_patch(Circle((x, y+6.28), radius=radius, color=color,
                 label=None, alpha=alpha, fill=fill))


def plot_events(LHE_list, jet_list, Jets, GenJets,label_list,index_list,save=None):
    #lab_list = ["bLept", "bHad", "Wc", "Wb"]
    color_list = ["green", "coral", "blue", "fuchsia"]

    for ev in index_list:
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(Jets[ev])):
            lab = "Jet" if i == 0 else None
            circle(ax, Jets.eta[ev, i],
                Jets.phi[ev, i], "red", label=lab)

        for i in range(len(GenJets[ev])):
            lab = "GenJet" if i == 0 else None
            circle(ax, GenJets.eta[ev, i],
                GenJets.phi[ev, i], "blue", label=lab)

        for i in range(len(LHE_list)):
            ax.plot(LHE_list[i][ev].eta, LHE_list[i][ev].phi, ".", color=color_list[i],
                    markersize=12, label=f"{label_list[i]}_LHE", alpha=1)
            circle(ax, jet_list[i][ev].eta, jet_list[i][ev].phi, color_list[i],
                label=f"{label_list[i]}_Jet", fill=True, alpha=0.3)

        plt.title(f"Jet-GenJet matching: ev {ev}")
        plt.xlim(-6, 6)
        plt.ylim(-3.14, 3.14)
        plt.grid(ls="--")
        plt.xlabel("$\eta$")
        plt.ylabel("$\phi$")
        plt.legend(bbox_to_anchor=(1.21, 1.))
        plt.subplots_adjust(left=0.1, right=0.75, top=0.85, bottom=0.15)

        if save is str:
            plt.savefig(f"{save}_{ev}.png")
        elif save==True:
            plt.savefig(f"./images/same_match_ev_{ev}.png")
        else:
            plt.show()
