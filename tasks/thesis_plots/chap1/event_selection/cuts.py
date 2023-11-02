#%%
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import awkward as ak


sys.path.append("../../")
sys.path.append("../../../../utils/coffea_utils")

from events import *
procs= list(x(sorted(n_ev.keys(), key=lambda k: n_ev[k])))


#%%
for proc in procs:
    print(f"Skimming {proc}...")
    datasets[proc]=obj_selection(datasets[proc])
    datasets[proc]["JetB"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavB,ascending=False),4)
    if proc not in ["tq","TTdiHad"]:
        datasets[proc]["JetC"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavB*(datasets[proc].Jet.btagDeepFlavCvB/(1-datasets[proc].Jet.btagDeepFlavCvB)),ascending=False),4)
    else:
        datasets[proc]["JetC"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavC,ascending=False),4)
        
        


#%%
def plot_func(datasets,plots,savepath,weights):
    for plot in plots:
        print("Plotting",plot,"...")
        fig,ax=plt.subplots()
        plots[plot]["figax"]=(fig,ax)
        fun=plots[plot]["fun"]
        bins=plots[plot]["bins"]
        
        to_plot=[]
        w_list=[]
        color_list=[]
        label_list=[]
        for proc in reversed(procs):
            print(proc)
            to_plot.append(fun(datasets[proc],proc))
            w_list.append(weights[proc])
            color_list.append(colors[proc])
            label_list.append(proc)

        ax.hist(to_plot,bins=bins,stacked=True,histtype="stepfilled",label=label_list,color=color_list,weights=w_list)

        ax.grid()
        ax.set_yscale("log")
        ax.set_axisbelow(True)
        ax.set_ylim(10,plots[plot]["ymax"])
        ax.legend()
        ax.set_xlabel(plots[plot]["xlabel"])
        ax.text(0.05,0.95,plots[plot]["text"],transform=ax.transAxes,va="top",ha="left")
        try:
            xticklabel=plots[plot]["labels"]
            ax.set_xticks(bins[:-1]+0.5)
            ax.set_xticklabels(xticklabel)
        except KeyError:
            pass
        
        
        mplhep.cms.text("Private Work",ax=ax)
        mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax)
        plt.savefig(savepath+plot+".png",bbox_inches='tight')

mplhep.style.use("CMS")
colors={"signal": "#011993",
        "TTsemiLept": "gold",
        "TTdiLept": "plum",
        "TTdiHad": "#5f9b8c",
        "WJets": "lightcoral",
        "WWJets":"#4376c9",
        "tW":"#00cccc",
        "tq":"#91d36e"
}




# %%
before_selection_plots={
    "nJet":{
        "fun":lambda ev,proc=None: ak.fill_none(ak.count(ev.Jet.pt,axis=1),0),
            "bins":np.linspace(-0.5,14.5,16),
            "labels":["0","","2","","4","","6","","8","","10","","12","","14"],
            "xlabel":"N Jets",
            "ymax":5e12,
            "text":"Inclusive Selection",
            },
    "nMuons":{
        "fun":lambda ev,proc=None: ak.fill_none(ak.count(ev.Muon.pt,axis=1),0),
        "bins":np.linspace(-0.5,4.5,6),
        "labels":["0","1","2","3","4"],
        "xlabel":"N Muons",
        "ymax":5e12,
        "text":"Inclusive Selection",
    },
    "nElectrons":{
        "fun":lambda ev,proc=None: ak.fill_none(ak.count(ev.Electron.pt,axis=1),0),
        "bins":np.linspace(-0.5,4.5,6),
        "labels":["0","1","2","3","4"],
        "xlabel":"N Electrons",
        "ymax":5e12,
        "text":"Inclusive Selection",
    },
}



# %%

plot_func(datasets,before_selection_plots,"plots/",weights)


#%%
#!EVENT SELECTION
#! fai prima un copy.copy
new_weights={}
new_datasets={}
for proc in procs:
    print(f"Skimming {proc}...")
    new_datasets[proc],new_weights[proc]=event_selection(datasets[proc],"both",weights[proc])

procs= list(reversed(sorted(n_ev.keys(), key=lambda k: sum(new_weights[k]))))
#%%
after_selection_plots={
    "Max_DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,0],
        "bins":np.linspace(0.,1,30),
        "xlabel":"max(DeepJetB)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "Second_Max_DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,1],
        "bins":np.linspace(0.,1,30),
        "xlabel":"2nd max(DeepJetB)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "Third_Max_DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,2],
        "bins":np.linspace(0.,1,30),
        "xlabel":"3rd max(DeepJetB)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "Fourth_Max_DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,3],
        "bins":np.linspace(0.,1,30),
        "xlabel":"4th max(DeepJetB)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "Max_DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,0],
        "bins":np.linspace(0.,1,30),
        "xlabel":"max(DeepJetC)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "Second_Max_DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,1],
        "bins":np.linspace(0.,1,30),
        "xlabel":"2nd max(DeepJetC)",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
    "NJet_DeepJetB_medium":{
        "fun":lambda ev,proc=None: ak.sum(ev.Jet.btagDeepFlavB>0.27,axis=1),
        "bins":np.linspace(-0.5,8.5,10),
        "xlabel":"N Jets medium bTag",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
        },
    "Leading_Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,0],
        "bins":np.linspace(20,500,30),
        "xlabel":"Leading Jet $p_T$ [GeV]",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
        },
    "Second_Leading_Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,1],
        "bins":np.linspace(20,450,27),
        "xlabel":"2nd Leading Jet $p_T$ [GeV]",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
        },
    "Third_Leading_Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,2],
        "bins":np.linspace(20,300,18),
        "xlabel":"3rd Leading Jet $p_T$ [GeV]",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
        },
    "Fourth_Leading_Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,3],
        "bins":np.linspace(20,250,15),
        "xlabel":"4th Leading Jet $p_T$ [GeV]",
        "ymax":5e10,
        "text":"$1\mu/e$+\n4 jets\nselection",
    },
}
# %%

plot_func(new_datasets,after_selection_plots,"plots/",new_weights)


#%%
#!Muon Selection
muon_plot={"Leading_Muon_pt":{
        "fun":lambda ev,proc=None: ev.Muon.pt,
        "bins":np.linspace(26,350,21),
        "xlabel":"Leading Muon $p_T$ [GeV]",
        "ymax":2.5e10,
        "text":"$1\mu$ +\n4 jets\nelection",
    },
}


muon_weights={}
muon_datasets={}
for proc in procs:
    print(f"Skimming {proc}...")
    muon_datasets[proc],muon_weights[proc]=event_selection(datasets[proc],"Muon",weights[proc])
    
#%%
plot_func(muon_datasets,muon_plot,"plots/",muon_weights)


#%%
#!Electron Selection
electron_plot={"Leading_Electron_pt":{
        "fun":lambda ev,proc=None: ev.Electron.pt,
        "bins":np.linspace(26,350,21),
        "xlabel":"Leading Electron $p_T$ [GeV]",
        "ymax":2.5e10,
        "text":"$1e$ +\n4 jets\nselection",
    },
}


electron_weights={}
electron_datasets={}
for proc in procs:
    print(f"Skimming {proc}...")
    electron_datasets[proc],electron_weights[proc]=event_selection(datasets[proc],"Electron",weights[proc])
    
#%%
plot_func(electron_datasets,electron_plot,"plots/",electron_weights)
# %%
