#%%Import and function definitions
import uproot
import matplotlib.pyplot as plt
import re
import numpy as np
import awkward as ak
import mplhep

mplhep.style.use("CMS")

signal = uproot.open("/scratchnvme/pviscone/Preselection_Skim/NN/predict/root/signal_predict_MuonCuts.root")["Events"]
background = uproot.open(
    "/scratchnvme/pviscone/Preselection_Skim/NN/predict/root/TTSemiLept_predict_MuonCuts.root")["Events"]


def rank(key):
    if bool(re.match("Muon_*", key)):
        key = f"{key}[:,0]"
    print(key)
    signal_array =(signal.arrays(key)[key])
    background_array =(background.arrays(key)[key])
    signal
    try:
        signal_array=ak.flatten(signal_array)
        background_array=ak.flatten(background_array)
    except:
        pass
    
    np.asarray(signal_array)[np.asarray(signal_array)==0]=None
    np.asarray(background_array)[np.asarray(background_array)==0]=None
    
    if True:
        bins=25
        plt.figure()
        fig,ax=plt.subplots(1,1)
        #ax.set_title(key)
        signal_not_nan=signal_array[~np.isnan(signal_array)]
        range_bin=(np.min(signal_not_nan), np.max(signal_not_nan))
        signal_hist = ax.hist(signal_array, bins=bins, range=range_bin,density=True,color="red",histtype="step",label="signalMu")[0]
        signal_hist = signal_hist/np.sum(signal_hist)
        background_hist = ax.hist(background_array, bins=bins, range=(
            np.min(signal_array), np.max(signal_array)),density=True,histtype="step",label="semileptMu")[0]
        background_hist = background_hist/np.sum(background_hist)
        plt.legend()
        ax.set_yscale("log")
        ax.set_xlabel(key)
        ax.set_ylabel("Density")
        ax.grid()
        fig.text(0.48,0.85,rf"$d={np.sum(np.abs(signal_hist-background_hist))*0.5:.2f}$")
        plt.show()
        return np.sum(np.abs(signal_hist-background_hist))*0.5
    else:
        return -1

#%% Rank


rank_dict = {}
for key in signal.keys():
    if bool(re.match("(Jet|Muon|MET)_*", key)):
        if bool(re.match("(Jet|Muon|MET)_phi", key)):
            continue
        ranking = rank(key)
        if ranking != -1:
            rank_dict[key] = ranking

#%%Plot
values = np.sort(np.array(list(rank_dict.values())))
keys = np.array(list(rank_dict.keys()))[
    np.argsort(np.array(list(rank_dict.values())))]

plt.figure(figsize=(5, 30))
plt.barh(np.arange(len(rank_dict)), values,
         height=0.5, tick_label=keys)
plt.xscale("log")
plt.grid()
plt.savefig("./images/ranking.png")
