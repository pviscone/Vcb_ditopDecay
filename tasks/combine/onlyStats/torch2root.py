#%%
import os
import numpy as np
import torch
import glob
import ctypes
old = os.environ.get("LD_LIBRARY_PATH")
os.environ["LD_LIBRARY_PATH"] = "/scratchnvme/pviscone/env/lib/:"+old
import ROOT

func= lambda x: np.arctanh(x)

def ptr(array):
    return (ctypes.c_double * len(array))(*array)



# %%
lumi=138e3
score_dict={"Muons":{},"Electrons":{}}
score_dict["Muons"]["path"]="/scratchnvme/pviscone/Preselection_Skim/NN/scores/"
score_dict["Electrons"]["path"]="/scratchnvme/pviscone/Preselection_Skim/NN_Electrons/scores/"

score_dict["Muons"]["weights"]={
    "signal":lumi*832*0.44*0.33*0.518*8.4e-4,
    "signalElectrons":lumi*832*0.44*0.33*0.0046*8.4e-4,
    "signalTaus":lumi*832*0.44*0.33*0.038*8.4e-4,
    "bkg":lumi*832*0.44*0.179*(1-8.4e-4),
    "diHad":lumi*832*0.45*0.0032,
    "diLept":lumi*832*0.11*0.2365,
    "WJets":lumi*59100*0.108*3*0.0003
}

score_dict["Electrons"]["weights"]={
    "signal":lumi*832*0.44*0.33*0.421*8.4e-4,
    "signalMuons":lumi*832*0.44*0.33*0.0016*8.4e-4,
    "signalTaus":lumi*832*0.44*0.33*0.027*8.4e-4,
    "bkg":lumi*832*0.44*0.144*(1-8.4e-4),
    "diHad":lumi*832*0.45*0.0012,
    "diLept":lumi*832*0.11*0.196,
    "WJets":lumi*59100*0.108*3*0.0003
}

score_dict["Muons"]["bins"]=np.concatenate((np.linspace(0,4.3,43),np.array([4.42,4.55,4.72,6.5])))
score_dict["Electrons"]["bins"]=np.linspace(0,5,30)




file=ROOT.TFile("hist2.root","RECREATE")

for key in score_dict:
    score_dict[key]["data"]={}
    for path in glob.glob(score_dict[key]["path"]+"*.pt"):
        name=path.split("/")[-1].split(".")[0]
        score_dict[key]["data"][name]=func(torch.load(path))
    directory=file.mkdir(key)
    directory.cd()
    for name in score_dict[key]["data"]:
        data=score_dict[key]["data"][name]
        one=np.ones(len(data))
        hist=ROOT.TH1F(name,name,len(score_dict[key]["bins"])-1,score_dict[key]["bins"])
        hist.Sumw2()
        hist.FillN(len(data),ptr(data),ptr(one*score_dict[key]["weights"][name.split("_")[0]]/len(data)))
        hist.Write()
    

file.Close()

#%%




"""
hist_dict={}
file=ROOT.TFile("hist.root","RECREATE")
directory=file.mkdir("hist")
directory.cd()
for key in score_dict:
    data=(score_dict[key])
    one=np.ones_like(data)
    hist_dict[key]=ROOT.TH1F(key,key,len(bins)-1,bins)
    hist_dict[key].Sumw2()
    hist_dict[key].FillN(len(data),ptr(data),ptr(one*weights[key.split("_")[0]]/len(data)))
    #hist_dict[key].FillN(len(data),ptr(data),ptr(one))
    hist_dict[key].Write()

file.Close()



c=ROOT.TCanvas()
hist_dict["signal_score_Muons"].Draw()
c.SetLogy()
c.Draw() """
