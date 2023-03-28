#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from matplotlib.patches import Circle

events = NanoEventsFactory.from_root(
    "../../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

def padded_matrix(ak_array,pad):
    masked_array = ak.pad_none(ak_array, pad, clip=True).to_numpy()
    masked_array.data[masked_array.mask] = 0
    return masked_array.data


def column(ak_array):
    return np.atleast_2d(ak_array.to_numpy(allow_missing=False)).T

def build_matrix(obj,variable_list):
    column_list=[]
    col_labels=[]
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

def pad_and_alternate(obj,variable_list,pad):
    matrix_list=[]
    col_labels=[]
    for var in variable_list:
        if var=="Tmass":
            matrix_list.append(padded_matrix(events.Tmass,pad))
        else:
            exec(f"matrix_list.append(padded_matrix(events.{obj}.{var},{pad}))")
    for i in range(pad):
        for var in variable_list:
            col_labels.append(f"{obj}{i}_{var}")
    return alternate_column(matrix_list), col_labels

def np_and(*args):
    res=args[0]
    for arg in args[1:]:
        res=np.bitwise_and(arg, res)
    return res

nu_pz = np.load("../../neutrino_reco/nu_pz.npy")
nu_pt = events.MET.pt.to_numpy()
nu_phi = events.MET.phi.to_numpy()
nu_eta = np.arcsinh(nu_pz/nu_pt)

nu_4Vect = ak.zip(
    {
        "pt": nu_pt,
        "eta": nu_eta,
        "phi": nu_phi,
        "mass": np.zeros_like(nu_pt),
    },
    with_name="PtEtaPhiMLorentzVector",
    behavior=vector.behavior,
)
del nu_pt, nu_eta, nu_phi, nu_pz



#%%

#* LHE Mask
#Select only the first product of the W decay
pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
#Mask for the leptonic decay of the W
leptonic_LHE_mask=np.bitwise_or(pdgId_Wdecay==13,pdgId_Wdecay==-13)
hadronic_LHE_mask=np.bitwise_not(leptonic_LHE_mask)

#*Define Jet objects

bLept_LHE = events.LHEPart[:, [2, 5]][leptonic_LHE_mask]
bHad_LHE = events.LHEPart[:, [2, 5]][hadronic_LHE_mask]
Wb_LHE = events.LHEPart[:, [4, 6]][hadronic_LHE_mask]
Wc_LHE = events.LHEPart[:, [3, 7]][hadronic_LHE_mask]


near_Jet,deltaR = events.LHEPart.nearest(events.Jet,return_metric=True)
bLept_Jet = near_Jet[:, [2, 5]][leptonic_LHE_mask]
bHad_Jet = near_Jet[:, [2, 5]][hadronic_LHE_mask]
Wb_Jet= near_Jet[:, [4, 6]][hadronic_LHE_mask]
Wc_Jet= near_Jet[:, [3, 7]][hadronic_LHE_mask]


#*Compute the deltaR between the LHE and the Jet
deltaRLept=deltaR[:, [2, 5]][leptonic_LHE_mask]
deltaRHad=deltaR[:, [2, 5]][hadronic_LHE_mask]
deltaRWb=deltaR[:, [4, 6]][hadronic_LHE_mask]
deltaRWc=deltaR[:, [3, 7]][hadronic_LHE_mask]


# * Apply the mask
Rmask = np_and(deltaRLept < 0.4,
              deltaRHad < 0.4,
              deltaRWb < 0.4,
              deltaRWc < 0.4,
              )

#*Compute the order of the jets in pt and compute the efficiency of selecting the first N

n_ev=len(bLept_Jet[Rmask])
bLept_pt_order = ak.argmin(events.Jet.delta_r(bLept_Jet), axis=1)
bHad_pt_order = ak.argmin(events.Jet.delta_r(bHad_Jet), axis=1)
Wb_pt_order = ak.argmin(events.Jet.delta_r(Wb_Jet), axis=1)
Wc_pt_order = ak.argmin(events.Jet.delta_r(Wc_Jet), axis=1)

plt.figure()

same_match = np.array([
    ak.sum(bLept_pt_order[Rmask] == bHad_pt_order[Rmask]),
    ak.sum(bLept_pt_order[Rmask] == Wb_pt_order[Rmask]),
    ak.sum(bLept_pt_order[Rmask] == Wc_pt_order[Rmask]),
    ak.sum(bHad_pt_order[Rmask] == Wb_pt_order[Rmask]),
    ak.sum(bHad_pt_order[Rmask] == Wc_pt_order[Rmask]),
    ak.sum(Wb_pt_order[Rmask] == Wc_pt_order[Rmask])
])
same_match=same_match/n_ev
lab=["bLept==bHad","bLept==Wb","bLept==Wc","bHad==Wb","bHad==Wc","Wb==Wc"]

plt.barh(lab,width=same_match)

same_match_matrix=np.array([
                    (bLept_pt_order[Rmask] == bHad_pt_order[Rmask]),
                    (bLept_pt_order[Rmask] == Wb_pt_order[Rmask]),
                    (bLept_pt_order[Rmask] == Wc_pt_order[Rmask]),
                    (bHad_pt_order[Rmask] == Wb_pt_order[Rmask]),
                    (bHad_pt_order[Rmask] == Wc_pt_order[Rmask]),
                    (Wb_pt_order[Rmask] == Wc_pt_order[Rmask])]).T


same_match_array = np.bitwise_or.reduce(same_match_matrix, axis=1)
same_match_idx = np.where(same_match_array)


print(f"Fraction of event with at least 2 jet matching to the same parton: {len(same_match_idx[0])/n_ev}")

assert ak.sum(bLept_pt_order==bHad_pt_order)==0
assert ak.sum(bLept_pt_order==Wb_pt_order)==0
assert ak.sum(bLept_pt_order==Wc_pt_order)==0
assert ak.sum(bHad_pt_order==Wb_pt_order)==0
assert ak.sum(bHad_pt_order==Wc_pt_order)==0
assert ak.sum(Wb_pt_order==Wc_pt_order)==0


n_max=15
plt.title("Jet pt order")
plt.hist(bLept_pt_order,label="bLept",histtype="step",range=(0,n_max),bins=n_max)
plt.hist(bHad_pt_order,label="bHad",histtype="step",range=(0,n_max),bins=n_max)
plt.yscale("log")
plt.xlabel("Jet Order in pt")
plt.legend()


num_jet_to_select=6


mask=np_and(Rmask,
            bLept_pt_order < num_jet_to_select,
            bHad_pt_order < num_jet_to_select,
            Wb_pt_order < num_jet_to_select,
            Wc_pt_order < num_jet_to_select,)


efficiency = ak.sum(mask)/ak.sum(Rmask)
print(f"Percentage of events with jets in the first {num_jet_to_select} jets:{efficiency}")





#%%
#*Mask the objects
events.Jet=events.Jet[:, :num_jet_to_select]
events = events[mask]
events.Neutrino = nu_4Vect[mask]

bLept_Jet = bLept_Jet[mask]
bHad_Jet = bHad_Jet[mask]
Wb_Jet = Wb_Jet[mask]
Wc_Jet = Wc_Jet[mask]

#*Compute some invariant masses
events.Muon = events.Muon[:, 0]
events.Neutrino.Wmass = (events.Muon+events.Neutrino).mass
events.Tmass = (events.Neutrino+events.Muon+events.Jet).mass


#%%

#*Build the matrix
col_labels=[]

muon_matrix,mu_labels=build_matrix("Muon",["pt","eta","phi","mass"])
nu_matrix,nu_labels=build_matrix("Neutrino",["pt","eta","phi","mass","Wmass"])
jet_matrix,jet_labels=pad_and_alternate("Jet",["pt","eta","phi","mass","btagDeepFlavB","btagDeepFlavCvB","btagDeepFlavCvL","area","Tmass"],num_jet_to_select)

bLept_label = column(bLept_pt_order[mask])
bHad_label=column(bHad_pt_order[mask])
Wb_label=column(Wb_pt_order[mask])
Wc_label=column(Wc_pt_order[mask])
col_labels=mu_labels+nu_labels+jet_labels+["bLept_label","bHad_label","Wb_label","Wc_label"]

event_matrix=np.hstack([muon_matrix,nu_matrix,jet_matrix,bLept_label,bHad_label,Wb_label,Wc_label])

event_df=pd.DataFrame(event_matrix,columns=col_labels)
#event_df.to_pickle("./event_df.pkl", compression="bz2")
# %%

#!Plot all the event which have multiple parton matching to the same jet
events.GenJet=events.GenJet[events.GenJet.pt>20]

lab_list = ["bLept", "bHad", "Wc", "Wb"]
color_list = ["green", "coral", "blue", "fuchsia"]


def circle(ax,x,y,color,label=None,fill=False,alpha=0.4,radius=0.4):
    ax.add_patch(Circle((x, y),radius=radius, color=color, label=label, alpha=alpha,fill=fill))
    ax.add_patch(Circle((x, y-6.28),radius=radius, color=color, label=None, alpha=alpha,fill=fill))
    ax.add_patch(Circle((x, y+6.28),radius=radius, color=color, label=None, alpha=alpha,fill=fill))

Jet_masked=events.Jet[Rmask]
GenJet_masked=events.GenJet[Rmask]
for ev in same_match_idx[0]:
    print(ev)

    LHE_list = [bLept_LHE[Rmask][ev], bHad_LHE[Rmask][ev], Wc_LHE[Rmask][ev], Wb_LHE[Rmask][ev]]
    jet_list = [bLept_Jet[Rmask][ev], bHad_Jet[Rmask][ev], Wc_Jet[Rmask][ev], Wb_Jet[Rmask][ev]]
        
    
    fig, ax = plt.subplots(figsize=(10,6))

    for i in range(len(Jet_masked[ev])):
        lab="Jet" if i ==0 else None
        circle(ax,Jet_masked.eta[ev, i],Jet_masked.phi[ev, i],"red",label=lab)
        
    for i in range(len(GenJet_masked[ev])):
        lab ="GenJet" if i ==0 else None
        circle(ax,GenJet_masked.eta[ev, i],GenJet_masked.phi[ev, i],"blue",label=lab)

    for i in range(len (LHE_list)):
        ax.plot(LHE_list[i].eta,LHE_list[i].phi,".",color=color_list[i],markersize=12, label=f"{lab_list[i]}_LHE",alpha=1)
        circle(ax, jet_list[i].eta, jet_list[i].phi, color_list[i], label=f"{lab_list[i]}_Jet", fill=True,alpha=0.3)

    plt.title(f"Jet-GenJet matching: ev {ev}")
    plt.xlim(-6, 6)
    plt.ylim(-3.14, 3.14)
    plt.grid(ls="--")
    plt.xlabel("$\eta$")
    plt.ylabel("$\phi$")
    plt.legend(bbox_to_anchor=(1.21, 1.))
    plt.subplots_adjust(left=0.1, right=0.75, top=0.85, bottom=0.15)
    plt.savefig(f"./images/same_match_ev_{ev}.png")
    #plt.show()
# %%
#!Inspect why the Wc match more often with the bLept than with the bHad
dR_bLept_Wc = bLept_LHE[Rmask].delta_r(Wc_LHE[Rmask])
dR_bHad_Wc = bHad_LHE[Rmask].delta_r(Wc_LHE[Rmask])
dR_bLept_Wb = bLept_LHE[Rmask].delta_r(Wb_LHE[Rmask])
dR_bHad_Wb = bHad_LHE[Rmask].delta_r(Wb_LHE[Rmask])

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.hist((dR_bLept_Wb), 50, alpha=0.7, label="$\Delta$R bLept-$W_b$")
plt.hist((dR_bHad_Wb), 50, alpha=0.7, label="$\Delta$R bHad-$W_b$")
plt.legend()
plt.xlabel("$\Delta R$")
plt.xlim(0, 4)
plt.subplot(1,2,2)
plt.hist((dR_bLept_Wc), 50, alpha=0.7, label="$\Delta$R bLept-$W_c$")
plt.hist((dR_bHad_Wc), 50, alpha=0.7, label="$\Delta$R bHad-$W_c$")
plt.legend()
plt.xlabel("$\Delta R$")
plt.xlim(0, 4)

print(
    f"Fraction of event with dR bLept-Wc<bHad-Wc {ak.sum(dR_bLept_Wc<dR_bHad_Wc)/len(dR_bLept_Wc)}")

print(
    f"Fraction of event with dR bLept-Wc<0.4 {len(dR_bLept_Wc[dR_bLept_Wc<0.4])/len(dR_bLept_Wc)}")


print(f"Fraction of event with dR bHad-Wc<0.4 {len(dR_bHad_Wc[dR_bHad_Wc<0.4])/len(dR_bHad_Wc)}")

print(f"mean dR bLept-Wc {ak.mean(dR_bLept_Wc)}")
print(f"mean dR bHad-Wc {ak.mean(dR_bHad_Wc)}")
print(f"mean dR bLept-Wc dR<0.4 {ak.mean(dR_bLept_Wc[dR_bLept_Wc<0.4])}")
print(f"mean dR bHad-Wc dR<0.4 {ak.mean(dR_bHad_Wc[dR_bHad_Wc<0.4])}")
