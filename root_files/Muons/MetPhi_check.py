#%%
import sys
sys.path.append("../../utils")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector
import awkward as ak
import mplhep
import importlib
import histogrammer
import pandas as pd
import ROOT


importlib.reload(histogrammer)
Histogrammer = histogrammer.Histogrammer


xkcd_yellow = mcolors.XKCD_COLORS["xkcd:golden yellow"]
mplhep.style.use(["CMS", "fira", "firamath"])


events = NanoEventsFactory.from_root(
    "TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root",
    schemaclass=NanoAODSchema,
).events()

#%%


plt.figure(figsize=(20,20))

plt.subplot(221)
h = Histogrammer(bins=50, histtype="step",linewidth=3.5,ylim=(1800, 3300),cmsText=None)
h.add_hist(events.GenMET.phi, label="GenMET_phi",histtype="stepfilled",color=xkcd_yellow)
h.add_hist(events.RawMET.phi, label="RawMET_phi", color="fuchsia", alpha=0.8)
h.add_hist(events.MET.phi,label="MET_phi",color="darkgreen",alpha=0.8)


mplhep.cms.text("Preliminary",loc=0)
h.plot()
plt.legend(bbox_to_anchor=(1.62, -0.25))
plt.xlabel("$\phi$")
plt.grid()
plt.subplot(222)

h.add_hist(events.GenMET.phi, label="GenMET_phi",
           histtype="stepfilled", color=xkcd_yellow)
h.add_hist(events.TkMET.phi,label="TkMET_phi",color="dodgerblue")
h.add_hist(events.CaloMET.phi,label="CaloMET_phi",color="black")

h.plot()
plt.legend(bbox_to_anchor=(0.9, -0.25))
plt.ylabel("")
plt.xlabel("$\phi$")
plt.grid()


plt.subplot(223)

h.add_hist(events.GenMET.phi, label="GenMET_phi",
           histtype="stepfilled", color=xkcd_yellow)

h.add_hist(events.ChsMET.phi, label="ChsMET_phi", color="firebrick", alpha=0.8)
h.add_hist(events.PuppiMET.phi, label="PuppiMET_phi", color="dimgray")
h.plot()

plt.xlabel("$\phi$")
plt.legend(bbox_to_anchor=(1.65, 0.55))
plt.grid()

#%%

dphi_raw=events.RawMET.delta_phi(events.GenMET)
dphi_met=events.MET.delta_phi(events.GenMET)
dphi_chs=events.ChsMET.delta_phi(events.GenMET)
dphi_tk=events.TkMET.delta_phi(events.GenMET)
dphi_calo=events.CaloMET.delta_phi(events.GenMET)
dphi_puppi=events.PuppiMET.delta_phi(events.GenMET)
plt.figure(figsize=(20, 10))

plt.subplot(121)
h = Histogrammer(bins=50, histtype="step", linewidth=3.5, cmsText=None,ylim=(0,14000),xlabel="$\Delta\phi_{GenMET}$")


h.add_hist(dphi_met, label="MET_phi", color=xkcd_yellow)
h.add_hist(dphi_raw, label="RawMET_phi", color="fuchsia", alpha=0.8)

h.add_hist(dphi_chs, label="ChsMET_phi", color="firebrick", alpha=0.8)
mplhep.cms.text("Preliminary", loc=0)
h.plot()
plt.legend(bbox_to_anchor=(2.62, 1.02))

plt.subplot(122)

h.add_hist(dphi_puppi, label="PuppiMET_phi", color="dimgray")

h.add_hist(dphi_calo, label="CaloMET_phi", color="black")
h.add_hist(dphi_tk, label="TkMET_phi", color="dodgerblue")
h.plot()
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.ylabel("")

#%%

def tprofile(dphi,phi=events.GenMET.phi,bins=50):
    tprof = ROOT.TProfile("", "", bins, -3.14, 3.14)
    for phi, dphi in zip(phi, dphi):
        tprof.Fill(phi, dphi, 1)
    return tprof


c=ROOT.TCanvas("","",1200,900)
c.SetTitle("TProfile")
c.Divide(3,2)
c.cd(1)
t_raw=tprofile(dphi_raw)
t_raw.SetTitle("$\Delta \Phi$GenMET- RawMET")
t_raw.GetXaxis().SetTitle("GenMET_phi")
t_raw.Draw()

c.cd(2)
t_met = tprofile(dphi_met)
t_met.SetTitle("$\Delta \Phi$GenMET- MET_phi")
t_met.GetXaxis().SetTitle("GenMET_phi")
t_met.Draw()

c.cd(3)
t_calo = tprofile(dphi_calo)
t_calo.SetTitle("$\Delta \Phi$GenMET- CaloMET")
t_calo.GetXaxis().SetTitle("GenMET_phi")
t_calo.Draw()

c.cd(4)
t_tk = tprofile(dphi_tk)
t_tk.SetTitle("$\Delta \Phi$GenMET- TkMET")
t_tk.GetXaxis().SetTitle("GenMET_phi")
t_tk.Draw()


c.cd(5)
t_chs = tprofile(dphi_chs)
t_chs.SetTitle("$\Delta \Phi$GenMET- ChsMET")
t_chs.GetXaxis().SetTitle("GenMET_phi")
t_chs.Draw()


c.cd(6)
t_puppi = tprofile(dphi_puppi)
t_puppi.SetTitle("$\Delta \Phi$GenMET- PuppiMET")
t_puppi.GetXaxis().SetTitle("GenMET_phi")
t_puppi.Draw()


c.Draw()




#%%
plt.figure()
h=Histogrammer(bins=50, histtype="step", linewidth=3.5, cmsText=None,ylim=(0,0.25),xlabel="$\phi$",density=True,ylabel="Density")
h.add_hist(ak.flatten(events.Jet.phi), label="Jet_phi", color=xkcd_yellow)
h.add_hist(ak.flatten(events.Muon.phi), label="Muon_phi", color="dodgerblue")
h.plot()
#%%

plt.hist2d(events.Muon.phi[:,0].to_numpy(),events.Muon.dxybs[:,0].to_numpy(),bins=50,range=[[-3.14,3.14],[-0.005,0.005]])
plt.xlabel("$\phi_{\mu}$")
plt.ylabel("$dxybs_{\mu}$")
plt.title("Muon_dxybs[0] vs Muon_phi[0]")

#%%
c = ROOT.TCanvas("", "",900,800)

t_raw = tprofile(events.Muon.dxybs[:,0],phi=events.Muon.phi[:,0],bins=50)
t_raw.SetTitle("TProfile Muon_dxybs[0] vs Muon_phi[0]")
t_raw.GetXaxis().SetTitle("Muon_phi[0]")
t_raw.GetYaxis().SetTitle("Muon_dxybs[0]")
t_raw.GetYaxis().SetRangeUser(-0.005,0.005)
t_raw.Draw()
c.Draw()
# %%
plt.figure(figsize=(10,15))
h = Histogrammer(bins=20,  linewidth=3.5, cmsText=None,density=True,xlabel="$\phi_{MET}$",ylim=(0.1,0.23),ylabel="Density")
plt.subplot(211)
h.add_hist(events.MET.phi[events.MET.pt < 20],
           label="MET_pt<20 [GeV]",color=xkcd_yellow,edgecolor="black")

h.add_hist(events.MET.phi[np.bitwise_and(events.MET.pt > 20, events.MET.pt < 50)],
           label="MET_pt $\in$(20,50) [GeV]",color="dodgerblue",edgecolor="blue",alpha=0.6)
mplhep.cms.text("Preliminary", loc=0)
h.plot()
plt.legend(bbox_to_anchor=(1.0, 0.9))
plt.grid()
plt.subplot(212)


h.add_hist(events.MET.phi[np.bitwise_and(events.MET.pt > 50, events.MET.pt < 100)],
           label="MET_pt $\in$(50,100) [GeV]", color=xkcd_yellow, edgecolor="black")

h.add_hist(events.MET.phi[events.MET.pt > 100], label="MET_pt>100 [GeV]",color="dodgerblue", edgecolor="blue", alpha=0.6)


plt.grid()
h.plot()
plt.legend(bbox_to_anchor=(1.0, 0.9))



#%%
import correctionlib

correction_labels = ["metphicorr_pfmet_mc","metphicorr_puppimet_mc"]


infile="./met.json"
corrected_pt = {}
corrected_phi = {}
for correction in correction_labels:
    ceval = correctionlib.CorrectionSet.from_file(infile)
    
    if "puppimet" in correction:
        pt = events.PuppiMET.pt
        phi = events.PuppiMET.phi
        key="Puppi"
    elif "pfmet" in correction:
        pt=events.MET.pt
        phi=events.MET.phi
        key="MET"
    
    npv = events.PV.npvs
    runs = None
    
    corrected_pt[key] = ceval[f"pt_{correction}"].evaluate(pt, phi, npv, runs)
    
    corrected_phi[key]=ceval[f"phi_{correction}"].evaluate(pt, phi, npv, runs)


corrected_puppi_phi=ak.Array(corrected_phi["Puppi"])
corrected_puppi_phi.phi=corrected_puppi_phi
corrected_met_phi=ak.Array(corrected_phi["MET"])
corrected_met_phi.phi=corrected_met_phi

plt.figure(figsize=(25,12))
h = Histogrammer(bins=40,  linewidth=3.5,xlabel="$\phi$",histtype="step")
plt.subplot(121)
h.add_hist(events.GenMET.phi,
           label="GenMet", edgecolor=xkcd_yellow,)

h.add_hist(events.PuppiMET.phi,
           label="PuppiMET", edgecolor="dodgerblue", )

h.add_hist(corrected_puppi_phi.phi,
           label="PuppiMET_corr", edgecolor="red", )
h.add_hist(events.MET.phi,label="MET",edgecolor="black", )

h.plot()
plt.ylim(2500, 4600)
plt.subplot(122)

h.add_hist(events.GenMET.delta_phi(events.PuppiMET),label="GenMet-PuppiMET",edgecolor=xkcd_yellow, )
h.add_hist(events.GenMET.delta_phi(corrected_puppi_phi),label="GenMet-PuppiMET_corr",edgecolor="red", )
h.add_hist(events.PuppiMET.delta_phi(corrected_puppi_phi),label="PuppiMET-PuppiMET_corr",edgecolor="dodgerblue", )
h.plot()
plt.ylim(1, 7000000)
plt.yscale("log")
plt.xlabel("$\Delta\phi$")


plt.figure(figsize=(25, 12))
h = Histogrammer(bins=40,  linewidth=3.5, xlabel="$\phi$", histtype="step")
plt.subplot(121)
h.add_hist(events.GenMET.phi,
           label="GenMet", edgecolor=xkcd_yellow,)

h.add_hist(events.MET.phi,
           label="MET", edgecolor="dodgerblue", )

h.add_hist(corrected_met_phi.phi,
           label="MET_corr", edgecolor="red", )


h.plot()
plt.ylim(2500, 4500)
plt.subplot(122)

h.add_hist(events.GenMET.delta_phi(events.MET),
           label="GenMet-MET", edgecolor=xkcd_yellow,bins=50 )
h.add_hist(events.GenMET.delta_phi(corrected_met_phi),
           label="GenMet-MET_corr", edgecolor="red", bins=50)
h.add_hist(events.MET.delta_phi(corrected_met_phi),
           label="MET-MET_corr", edgecolor="dodgerblue",bins=50 )
h.plot()
plt.ylim(300, 70000)
plt.yscale("log")
plt.xlabel("$\Delta\phi$")
