#%%Import and function definitions
import uproot
import matplotlib.pyplot as plt
import re
import numpy as np

signal = uproot.open(
    "../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root")["Events"]
background = uproot.open(
    "../TTbarSemileptonic_Nocb_MuonSelection.root")["Events"]


def rank(key):
    if bool(re.match("Muon_*", key)):
        key = f"{key}[:,0]"
    signal_array = signal.arrays(key, library="pd")
    background_array = background.arrays(key, library="pd")
    if signal_array[key].dtype != np.bool_:
        signal_hist = np.histogram(signal_array[key], bins=60, range=(
            np.min(signal_array[key]), np.max(signal_array[key])))[0]
        signal_hist = signal_hist/signal_hist.sum()
        background_hist = np.histogram(background_array[key], bins=60, range=(
            np.min(signal_array[key]), np.max(signal_array[key])))[0]
        background_hist = background_hist/background_hist.sum()
        return np.sum(np.abs(signal_hist-background_hist))*0.5
    else:
        return -1

#%% Rank


rank_dict = {}
for key in signal.keys():
    if bool(re.match("(Jet|Muon|MET)_*", key)):
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
