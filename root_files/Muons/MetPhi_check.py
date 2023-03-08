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


h.add_hist(dphi_met, label="GenMET_phi", color=xkcd_yellow)
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
t_raw.SetTitle("RawMET_phi")
t_raw.GetXaxis().SetTitle("GenMET_phi")
t_raw.Draw()

c.cd(2)
t_met = tprofile(dphi_met)
t_met.SetTitle("MET_phi")
t_met.GetXaxis().SetTitle("GenMET_phi")
t_met.Draw()

c.cd(3)
t_calo = tprofile(dphi_calo)
t_calo.SetTitle("CaloMET_phi")
t_calo.GetXaxis().SetTitle("GenMET_phi")
t_calo.Draw()

c.cd(4)
t_tk = tprofile(dphi_tk)
t_tk.SetTitle("TkMET_phi")
t_tk.GetXaxis().SetTitle("GenMET_phi")
t_tk.Draw()


c.cd(5)
t_chs = tprofile(dphi_chs)
t_chs.SetTitle("ChsMET_phi")
t_chs.GetXaxis().SetTitle("GenMET_phi")
t_chs.Draw()


c.cd(6)
t_puppi = tprofile(dphi_puppi)
t_puppi.SetTitle("PuppiMET_phi")
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