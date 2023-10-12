#%%
import sys
import matplotlib.pyplot as plt
import mplhep



sys.path.append("../")
sys.path.append("../../../utils/coffea_utils")

from events import *


# %%

procs= list(reversed(sorted(n_ev.keys(), key=lambda k: n_ev[k])))
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
            "bins":np.arange(0,30),
            },
    "DeepJetB":{
        "fun":lambda ev,proc=None: (ev.Jet.btagDeepFlavB),
        "bins":np.linspace(0,1,100),
        },
    "DeepJetC":{
        "fun":lambda ev,proc: (ev.Jet.btagDeepFlavB*(ev.Jet.btagDeepFlavCvB/(1-ev.Jet.btagDeepFlavCvB))) if proc not in ["tq","TTdiHad"] else (ev.Jet.btagDeepFlavC),
        "bins":np.linspace(0,1,100),
        },
    "LHE_quarkFlav":{
        "fun":lambda ev,proc=None: (ev.LHEPart.pdgId[np_and(np.abs(ev.LHEPart.pdgId)<6,ev.LHEPart.status==1)]),
        "bins":np.linspace(0.5,5.5,6),
        },
}

def plot_func(datasets,plots,prefix=""):
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
        mplhep.cms.text("Private Work",ax=ax)
        mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax)
        plt.savefig("plots/"+prefix+plot+".png")
# %%

plot_func(datasets,plots)