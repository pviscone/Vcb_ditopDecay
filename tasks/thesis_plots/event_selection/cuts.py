#%%
import sys
import matplotlib.pyplot as plt
import mplhep



sys.path.append("../")
sys.path.append("../../../utils/coffea_utils")

from events import *
procs= list(reversed(sorted(n_ev.keys(), key=lambda k: n_ev[k])))

def plot_func(datasets,plots,savepath):
    for plot in plots:
        print("Plotting",plot,"...")
        fig,ax=plt.subplots(figsize=(10,8))
        plots[plot]["figax"]=(fig,ax)
        fun=plots[plot]["fun"]
        bins=plots[plot]["bins"]
        for proc in procs:
            print(proc)
            to_plot=fun(datasets[proc],proc)
            
            if to_plot.layout.purelist_depth>1:
                w=ak.flatten(ak.ones_like(to_plot)*weights[proc])
                to_plot=ak.flatten(to_plot)

            else:
                w=weights[proc]
            
            
            ax.hist(to_plot,bins=bins,stacked=True,histtype="stepfilled",label=proc,color=colors[proc],weights=w)

        ax.grid()
        ax.set_yscale("log")
        ax.set_axisbelow(True)
        ax.set_ylim(10,3e12)
        ax.legend(fontsize=14)
        ax.set_xlabel(plot)
        try:
            xticklabel=plots[plot]["labels"]
            ax.set_xticks(bins[:-1]+0.5)
            ax.set_xticklabels(xticklabel)
        except KeyError:
            pass
        
        
        mplhep.cms.text("Private Work",ax=ax)
        mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax)
        plt.savefig(savepath+plot+".png")



for proc in procs:
    datasets[proc]["JetB"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavB,ascending=False),4)
    if proc not in ["tq","TTdiHad"]:
        datasets[proc]["JetC"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavB*(datasets[proc].Jet.btagDeepFlavCvB/(1-datasets[proc].Jet.btagDeepFlavCvB)),ascending=False),4)
    else:
        datasets[proc]["JetC"]=ak.pad_none(ak.sort(datasets[proc].Jet.btagDeepFlavC,ascending=False),4)
        
        
        
        

# %%


mplhep.style.use("CMS")
colors={"signal": "#011993",
        "TTsemiLept": "gold",
        "TTdiLept": "plum",
        "TTdiHad": "limegreen",
        "WJets": "lightcoral",
        "WWJets":"#4376c9",
        "tW":"#00cccc",
        "tq":"#91d36e"
}




plots={
    "nJet":{
        "fun":lambda ev,proc=None: ak.num(ev.Jet),
            "bins":np.linspace(0,25,20),
            },
    "DeepJetB":{
        "fun":lambda ev,proc=None: (ev.Jet.btagDeepFlavB),
        "bins":np.linspace(0,1,30),
        },
    "Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,0],
        "bins":np.linspace(0,1,30),
    },
    "Second Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,1],
        "bins":np.linspace(0,1,30),
    },
    "Third Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,2],
        "bins":np.linspace(0,1,30),
    },
    "Fourth Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,3],
        "bins":np.linspace(0,1,30),
    },
    "Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,0],
        "bins":np.linspace(0,1,30),
    },
    "Second Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,1],
        "bins":np.linspace(0,1,30),
    },
    "Third Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,2],
        "bins":np.linspace(0,1,30),
    },
    "Fourth Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,3],
        "bins":np.linspace(0,1,30),
    },
    "DeepJetC":{
        "fun":lambda ev,proc: ev.JetC,
        "bins":np.linspace(0,1,30),
        },
    "LHE_quarkFlav":{
        "fun":lambda ev,proc=None: (ev.LHEPart.pdgId[np_and(np.abs(ev.LHEPart.pdgId)<6,ev.LHEPart.status==1)]),
        "bins":np.linspace(0.5,5.5,6),
        "labels":["d","u","s","c","b"],
        },
    "LHE_#b":{
        "fun":lambda ev,proc=None: ak.sum(np.abs(ev.LHEPart.pdgId[ev.LHEPart.status==1])==5,axis=1),
        "bins":np.linspace(-0.5,6.5,8),
        },
    "LHE_#c":{
        "fun":lambda ev,proc=None: ak.sum(np.abs(ev.LHEPart.pdgId[ev.LHEPart.status==1])==4,axis=1),
        "bins":np.linspace(-0.5,6.5,8),
        },
    "#Jet_DeepJetB>0.27":{
        "fun":lambda ev,proc=None: ak.sum(ev.Jet.btagDeepFlavB>0.27,axis=1),
        "bins":np.linspace(-0.5,8.5,10),
        },
    "Jet_pt":{
        "fun":lambda ev,proc=None: ev.Jet.pt,
        "bins":np.linspace(0,500,30),
        },
    "Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,0],
        "bins":np.linspace(0,500,30),
        },
    "Second Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,1],
        "bins":np.linspace(0,400,24),
        },
    "Third Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,2],
        "bins":np.linspace(0,350,21),
        },
    "Fourth Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,3],
        "bins":np.linspace(0,300,18),
    },
    "Leading Lepton_pt":{
        "fun":lambda ev,proc=None: ak.max(ak.concatenate([ev.Electron.pt,ev.Muon.pt],axis=1),axis=1),
        "bins":np.linspace(0,400,24),
    },
}


# %%

plot_func(datasets,plots,"plots/no_selections/")


#%%
#!OBJECT SELECTIONS
selected_obj={}
for proc in procs:
    selected_obj[proc]=obj_selection(datasets[proc])
    selected_obj[proc]["JetB"]=ak.pad_none(ak.sort(selected_obj[proc].Jet.btagDeepFlavB,ascending=False),4)
    if proc not in ["tq","TTdiHad"]:
        selected_obj[proc]["JetC"]=ak.pad_none(ak.sort(selected_obj[proc].Jet.btagDeepFlavB*(selected_obj[proc].Jet.btagDeepFlavCvB/(1-selected_obj[proc].Jet.btagDeepFlavCvB)),ascending=False),4)
    else:
        selected_obj[proc]["JetC"]=ak.pad_none(ak.sort(selected_obj[proc].Jet.btagDeepFlavC,ascending=False),4)
    
    
    
obj_plots={
    "nJet":{
        "fun":lambda ev,proc=None: ak.num(ev.Jet),
            "bins":np.linspace(-0.5,10.5,12),
            },
    "DeepJetB":{
        "fun":lambda ev,proc=None: (ev.Jet.btagDeepFlavB),
        "bins":np.linspace(0,1,30),
        },
    "Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,0],
        "bins":np.linspace(0,1,30),
    },
    "Second Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,1],
        "bins":np.linspace(0,1,30),
    },
    "Third Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,2],
        "bins":np.linspace(0,1,30),
    },
    "Fourth Max DeepJetB":{
        "fun":lambda ev,proc=None: ev.JetB[:,3],
        "bins":np.linspace(0,1,30),
    },
    "Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,0],
        "bins":np.linspace(0,1,30),
    },
    "Second Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,1],
        "bins":np.linspace(0,1,30),
    },
    "Third Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,2],
        "bins":np.linspace(0,1,30),
    },
    "Fourth Max DeepJetC":{
        "fun":lambda ev,proc=None: ev.JetC[:,3],
        "bins":np.linspace(0,1,30),
    },
    "DeepJetC":{
        "fun":lambda ev,proc: ev.JetC,
        "bins":np.linspace(0,1,30),
        },
    "#Jet_DeepJetB>0.27":{
        "fun":lambda ev,proc=None: ak.sum(ev.Jet.btagDeepFlavB>0.27,axis=1),
        "bins":np.linspace(-0.5,8.5,10),
        },
    "Jet_pt":{
        "fun":lambda ev,proc=None: ev.Jet.pt,
        "bins":np.linspace(0,500,30),
        },
    "Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,0],
        "bins":np.linspace(0,500,30),
        },
    "Second Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,1],
        "bins":np.linspace(0,400,24),
        },
    "Third Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,2],
        "bins":np.linspace(0,350,21),
        },
    "Fourth Leading Jet_pt":{
        "fun":lambda ev,proc=None: ak.pad_none(ev.Jet.pt,4)[:,3],
        "bins":np.linspace(0,000,18),
    },
    "Leading Lepton_pt":{
        "fun":lambda ev,proc=None: ak.max(ak.concatenate([ev.Electron.pt,ev.Muon.pt],axis=1),axis=1),
        "bins":np.linspace(0,400,24),
    },
}
plot_func(selected_obj,plots,"plots/obj_selections/")


#%%
#!EVENT SELECTIONS