#%% Imports
import sys
sys.path.append("../../../utils")
import mplhep
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from selections import obj_selection,mu_selection

xkcd_yellow = mcolors.XKCD_COLORS["xkcd:golden yellow"]
mplhep.style.use(["CMS", "fira", "firamath"])
class Jets():
    def __init__(self,*args):
        self.jet_list=args
    
    def get(self,what):
        res=[]
        for jet in self.jet_list:
            res.append(getattr(jet,what))
        return res

    def get_methods(self,method,*args):
        res=[]
        for jet in self.jet_list:
            res.append(getattr(jet,method)(*args))
        return res
    
import copy
def stacked(list_plot,
            bins=None,
            label=None,
            ylabel="",
            xlabel="",
            units="",
            savefig=False,
            yfactor=1.2,
            log=False,
            legend_loc="best",
            density=False,
            colors=["dodgerblue",xkcd_yellow],):
    
    plt.rc("font", size=30)
    plt.rc('legend', fontsize=22)
    plt.figure()
    label_to_plot=copy.copy(label)
    for i in range(len(list_plot)):
        label_to_plot[i]=label[i]+f"\n{np.mean(list_plot[i]):.2f}({np.std(list_plot[i]):.2f}) {units}"
    
    h=plt.hist(list_plot,stacked=True, color=colors,label=label_to_plot,bins=bins,density=density)
    plt.grid()
    plt.legend(loc=legend_loc)
    plt.ylabel(ylabel)
    plt.ylim(0,np.max(h[0][-1])*yfactor)
    mplhep.cms.text("Private Work",loc=2)
    plt.xlabel(xlabel)
    plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    if log:
        plt.yscale("log")
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    return h

def np_and(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.bitwise_and(res, arg)
    return res




# I don't know why but the first plot (only) is always weird.
# So,as a workaround, these two lines create a dummy plot.
plt.hist([0])
plt.close()

events = NanoEventsFactory.from_root(
    "/scratchnvme/pviscone/Preselection_Skim/signal/signal_train.root",
    schemaclass=NanoAODSchema,
).events()

events=obj_selection(events)
events=mu_selection(events)

#%%

events_mu=events[ak.any(events.LHEPart.pdgId==13,axis=1)]
mu_lhe=events_mu.LHEPart[:,6]
b_had_mu=events_mu.LHEPart[:,2]
b_lept_mu=events_mu.LHEPart[:,5]
c_W_mu=events_mu.LHEPart[:,3]
b_W_mu=events_mu.LHEPart[:,4]




events_amu=events[ak.any(events.LHEPart.pdgId==-13,axis=1)]
amu_lhe=events_amu.LHEPart[:,3]
b_had_amu=events_amu.LHEPart[:,5]
b_lept_amu=events_amu.LHEPart[:,2]
c_W_amu=events_amu.LHEPart[:,7]
b_W_amu=events_amu.LHEPart[:,6]



signal=ak.concatenate([events_mu,events_amu])
mu_lhe=ak.concatenate([mu_lhe,amu_lhe])
b_had_lhe=ak.concatenate([b_had_mu,b_had_amu])
b_lept_lhe=ak.concatenate([b_lept_mu,b_lept_amu])
c_W_lhe=ak.concatenate([c_W_mu,c_W_amu])
b_W_lhe=ak.concatenate([b_W_mu,b_W_amu])


#%%


signal["bHad"]=ak.singletons(b_had_lhe).nearest(signal.Jet,threshold=0.4)
signal["bLept"]=ak.singletons(b_lept_lhe).nearest(signal.Jet,threshold=0.4)
signal["bW"]=ak.singletons(b_W_lhe).nearest(signal.Jet,threshold=0.4)
signal["cW"]=ak.singletons(c_W_lhe).nearest(signal.Jet,threshold=0.4)

#%%
signal=signal[np_and(ak.count(signal["bHad"].pt,axis=1)==1,
                     ak.count(signal["bLept"].pt,axis=1)==1,
                     ak.count(signal["bW"].pt,axis=1)==1,
                     ak.count(signal["cW"].pt,axis=1)==1,)]

signal=signal[np_and(signal["bHad"].pt!=signal["bLept"].pt,
                     signal["bHad"].pt!=signal["bW"].pt,
                     signal["bHad"].pt!=signal["cW"].pt,
                     signal["bLept"].pt!=signal["bW"].pt,
                     signal["bLept"].pt!=signal["cW"].pt,
                     signal["bW"].pt!=signal["cW"].pt,)[:,0]]

bHad=signal["bHad"][:,0]
bLept=signal["bLept"][:,0]
bW=signal["bW"][:,0]
cW=signal["cW"][:,0]


others=signal.Jet[np_and(signal.Jet.pt!=bHad.pt,
                          signal.Jet.pt!=bLept.pt,
                          signal.Jet.pt!=bW.pt,
                          signal.Jet.pt!=cW.pt)]

jets=Jets(bHad,bLept,bW,cW,ak.flatten(others))
label=["bHad","bLept","bW","cW","others"]

#%%
bHad_idx=ak.argmax(signal.Jet.pt==bHad.pt,axis=1)
bLept_idx=ak.argmax(signal.Jet.pt==bLept.pt,axis=1)
bW_idx=ak.argmax(signal.Jet.pt==bW.pt,axis=1)
cW_idx=ak.argmax(signal.Jet.pt==cW.pt,axis=1)

#%%
stacked([bHad_idx,bLept_idx,bW_idx,cW_idx],
        colors=["dodgerblue",xkcd_yellow,"darkorange","limegreen"],
        label=["bHad","bLept","bW","cW"],
        bins=np.linspace(-0.5,8.5,10),
        xlabel="$p_T$ index",
        savefig="plots/pt_index.png",
)
        
        




#%%
colors=["dodgerblue",xkcd_yellow,"darkorange","limegreen","lightsteelblue"]
stacked(jets.get("pt"),
        label=label,
        colors=colors,
        bins=np.linspace(0,300,50),
        density=False,
        xlabel=r"$p_T [GeV]$",
        units="[GeV]",
        savefig="plots/jet_pt.png")


stacked(jets.get("eta"),
        label=label,
        colors=colors,
        bins=np.linspace(-5,5,50),
        density=False,
        xlabel=r"$\eta$",
        savefig="plots/jet_eta.png")



stacked(jets.get("btagDeepFlavB"),
        label=label,
        colors=colors,
        bins=np.linspace(0,1,50),
        density=False,
        xlabel=r"DeepJetB",
        savefig="plots/jet_b.png",
        legend_loc='upper center')


stacked(jets.get("btagDeepFlavCvB"),
        label=label,
        colors=colors,
        bins=np.linspace(0,1,50),
        density=False,
        xlabel=r"DeepJetCvB",
        savefig="plots/jet_cvb.png")

stacked(jets.get("btagDeepFlavCvL"),
        label=label,
        colors=colors,
        bins=np.linspace(0,1,50),
        density=False,
        xlabel=r"DeepJetCvL",
        savefig="plots/jet_cvl.png",
        legend_loc='upper center')

""" stacked(jets.get_methods("delta_r",signal.Muon[:,0]),
        label=label,
        colors=colors,
        bins=np.linspace(0,5,50),
        density=False,
        xlabel=r"$\Delta R (j-\mu)$",
        savefig=False) """


#%%
