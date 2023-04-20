#%%
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep

sys.path.append("../../../utils/coffea_utils")
from coffea_utils import Electron_cuts, Muon_cuts, Jet_parton_matching,np_and,np_or


#%%
#! REMEMBER TO VOMS-PROXY-INIT BEFORE RUNNING THIS SCRIPT
TTJets_diLept = NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2820000/C6DC1D11-72BB-8C45-8625-0B874FB177C2.root",
    schemaclass=NanoAODSchema
).events()


TTJets_diHad = NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2520000/3D7C0328-DC6C-C046-8DF4-C7024E774FBA.root",
    schemaclass=NanoAODSchema
).events()

TTJets_diHad = TTJets_diHad[
    np_and(
        np.abs(TTJets_diHad.GenPart[TTJets_diHad.GenPart.pdgId==-24][:,-1].children.pdgId[:,0])<6,
        np.abs(TTJets_diHad.GenPart[TTJets_diHad.GenPart.pdgId==24][:,-1].children.pdgId[:,0])<6
    )
]


WJets_toLNu=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv2/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/230000/87A6BF8E-A6CC-0549-ADC0-260011EE2894.root",
    schemaclass=NanoAODSchema
).events()

#!There are negative genweights in this sample
WWJets_LNuQQ=NanoEventsFactory.from_root(
    "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/60000/43340426-80F1-534C-9D61-5F0D90AD57B3.root",
    schemaclass=NanoAODSchema
).events()

TT_semilept_cb=NanoEventsFactory.from_root(
    "../../../root_files/generated_signal/TTbarSemileptonic_cbOnly_pruned_optimized.root",
    schemaclass=NanoAODSchema
).events()


TT_semilept_cb_Electron=TT_semilept_cb[
    np_or(np.abs(TT_semilept_cb.LHEPart.pdgId[:,3]==11),
          np.abs(TT_semilept_cb.LHEPart.pdgId[:,6]==11))
    ]
TT_semilept_cb_Muon=TT_semilept_cb[
    np_or(np.abs(TT_semilept_cb.LHEPart.pdgId[:,3]==13),
          np.abs(TT_semilept_cb.LHEPart.pdgId[:,6]==13))
    ]
TT_semilept_cb_Tau=TT_semilept_cb[
    np_or(np.abs(TT_semilept_cb.LHEPart.pdgId[:,3]==15),
          np.abs(TT_semilept_cb.LHEPart.pdgId[:,6]==15))
    ]

del TT_semilept_cb

TT_Jets_LNuQQ_NoCKM=NanoEventsFactory.from_root(
    "../../../root_files/Skimmed_background/results/TTbarSemileptonic_Nocb.root",
    schemaclass=NanoAODSchema
).events()

TT_Jets_LNuQQ_NoCKM_Electron=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==11),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==11))
    ]
TT_Jets_LNuQQ_NoCKM_Muon=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==13),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==13))
    ]
TT_Jets_LNuQQ_NoCKM_Tau=TT_Jets_LNuQQ_NoCKM[
    np_or(np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,3]==15),
          np.abs(TT_Jets_LNuQQ_NoCKM.LHEPart.pdgId[:,6]==15))
    ]

del TT_Jets_LNuQQ_NoCKM


datasets = {
    "TTJets_cbENu": TT_semilept_cb_Electron,
    "TTJets_cbMuNu": TT_semilept_cb_Muon,
    "TTJets_cbTauNu": TT_semilept_cb_Tau,
    "TTJets_LNuQQ_NoCKM_E": TT_Jets_LNuQQ_NoCKM_Electron,
    "TTJets_LNuQQ_NoCKM_Mu": TT_Jets_LNuQQ_NoCKM_Muon,
    "TTJets_LNuQQ_NoCKM_Tau": TT_Jets_LNuQQ_NoCKM_Tau,
    "TTJets_diLept": TTJets_diLept,
    "TTJets_diHad": TTJets_diHad,
    "WJets_toLNu": WJets_toLNu,
    "WWJets_LNuQQ":WWJets_LNuQQ
}



#%%
"""
%%HTML
<style type="text/css">
table.dataframe td, table.dataframe th {
    border: 1px  black solid !important;
  color: black !important;
}
</style>
"""

Muon_selector=Muon_cuts()
Electron_selector=Electron_cuts()
Muon_cuts_dict={}
Electron_cuts_dict={}
for dataset in datasets:
    print(f"@@@@@@@@@@@@@@@@@@@@@@@ {dataset} @@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("==========================Muon cuts==============================")
    Muon_cuts_dict[dataset]=Muon_selector.process(datasets[dataset],out="cuts")
    
    print("\n========================Electron cuts============================")
    Electron_cuts_dict[dataset]=Electron_selector.process(datasets[dataset],out="cuts")
    print("\n\n")
# %%
pd.set_option('display.precision', 3)

def fill_dataframe(cuts_dict,datasets):
    cuts_list=[cut["cut"] for cut in cuts_dict[list(datasets.keys())[0]]]
    for idx_dataset,dataset in enumerate(datasets):
        cuts=[]
        eff=[]
        typ=[]
        eff=[]
        for idx,cut in enumerate(cuts_list):
            cuts.append(cut)
            cuts.append(cut)
            typ.append("relative")
            eff.append(cuts_dict[dataset][idx]["relative"])
            typ.append("cumulative")
            eff.append(cuts_dict[dataset][idx]["cumulative"])
        data_dict={"cut":cuts,"type":typ,dataset:eff}
        df=pd.DataFrame(data_dict)

        if idx_dataset==0:
            final_df=df
        else:
            final_df=pd.concat([final_df,df[dataset]],axis=1)
    final_df=final_df.set_index(["cut","type"])
    final_df=final_df.T.stack()
    final_df=final_df[cuts_list]
    final_df.style.set_properties(**{'text-align': 'center'})
    final_df.style.set_properties(**{'border': '1.3px solid black',
                          'color': 'black'})
    return final_df

muon_df=fill_dataframe(Muon_cuts_dict,datasets)
electron_df=fill_dataframe(Electron_cuts_dict,datasets)

muon_report=muon_df.groupby(
    "type").apply(lambda x: x).style.background_gradient()
electron_report=electron_df.groupby(
    "type").apply(lambda x: x).style.background_gradient()
i=0

#.groupby("type").apply(lambda x: x)
#.style.background_gradient()
#.xs("cumulative",level=1)

#%%
#BF
#tt fully had = 0.478864
#tt semilept=   0.007058
#tt dilept=     0.010404
#W leptonic=    0.102

#cross
#tt fully had = 832
#W leptonic=    59100
#WW leptonic=   119

#|vcb|=0.041
lumi=138*1e3
num_events={
    "TTJets": 832,
    "WJets_leptonic":19700*3,
    "WWJets":119#(multiply for sempilemptonic bf)
}



num_events={"TTJets_cbENu":lumi*832*0.478864*8.4e-4/3,
            "TTJets_cbMuNu":lumi*832*0.478864*8.4e-4/3,
            "TTJets_cbTauNu":lumi*832*0.478864*8.4e-4/3,
            "TTJets_LNuQQ_NoCKM_E":lumi*832*0.478864*(1-8.4e-4)/3,
            "TTJets_LNuQQ_NoCKM_Mu":lumi*832*0.478864*(1-8.4e-4)/3,
            "TTJets_LNuQQ_NoCKM_Tau":lumi*832*0.478864*(1-8.4e-4)/3,
            "TTJets_diLept":lumi*832*0.010404,
            "TTJets_diHad":lumi*832*0.478864,
            "WJets_toLNu":lumi*59100*0.102,
            "WWJets_LNuQQ":lumi*119*0.007058
}

muon_cuts_events=muon_df.xs("cumulative",level=1)
muon_cuts_events.insert(0,"Total Run2",0)
electron_cuts_events=electron_df.xs("cumulative",level=1)
electron_cuts_events.insert(0,"Total Run2",0)


for key in num_events:
    muon_cuts_events.loc[key,"Total Run2"]=num_events[key]
    electron_cuts_events.loc[key,"Total Run2"]=num_events[key]
    muon_cuts_events.loc[key]=muon_cuts_events.loc[key]*num_events[key]
    electron_cuts_events.loc[key]=electron_cuts_events.loc[key]*num_events[key]
    
    
plt.figure(figsize=(12,5))
plt.subplot(121)
muon_cuts_events["Total Run2"].plot(
    kind="barh",label="Total",color="dodgerblue")
muon_cuts_events[muon_cuts_events.columns[-1]].plot(
    kind="barh",label="After cuts",color="gold")
plt.title("Muon cuts")
plt.xscale("log")
plt.grid(linestyle=":",alpha=0.8)
plt.xlabel("Events")
plt.legend()
mplhep.cms.text("Private Work")


plt.subplot(122)

electron_cuts_events["Total Run2"].plot(
    kind="barh",label="Total",color="dodgerblue")
electron_cuts_events[electron_cuts_events.columns[-1]].plot(
    kind="barh",label="After cuts",color="gold")

plt.tick_params(axis='y', labelleft=False, labelright=False)
plt.title("Electron cuts")
plt.xscale("log")
plt.grid(linestyle=":",alpha=0.8)
plt.xlabel("Events")
plt.legend()
plt.tight_layout()
mplhep.cms.lumitext("$138$ $fb^{-1}$")
#%%