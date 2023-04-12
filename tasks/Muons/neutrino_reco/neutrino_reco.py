#%% Includes and function definitions
#%load_ext autoreload
#%autoreload 2
from coffea.nanoevents.methods import vector
import sys

sys.path.append("../../../utils")

import uproot
import ROOT
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from histogrammer import Histogrammer
import mplhep as hep
import matplotlib as mpl
import matplotlib.colors as mcolors
import awkward as ak



plt.style.use(hep.style.CMS)
signal = uproot.open("../TTbarSemileptonic_cbOnly_pruned_optimized_MuonSelection.root")["Events"]
#signal=uproot.open("../BigMuon_MuonSelection.root")["Events"]
#background = uproot.open("../TTbarSemileptonic_Nocb_MuonSelection.root")["Events"]


def get(key, numpy=True, library="pd"):
    arr = signal.arrays(key, library=library)[key]
    if numpy == True:
        return arr.to_numpy()
    else:
        return arr


def neutrino_pz(lept_pt, lept_eta, lept_phi, MET_pt, MET_phi):
    Mw = 80.385
    El2 = lept_pt**2*np.cosh(lept_eta)**2
    Pt_scalar_product = MET_pt*lept_pt*np.cos(MET_phi-lept_phi)
    a = lept_pt**2
    b = -lept_pt*np.sinh(lept_eta)*(Mw**2+2*Pt_scalar_product)
    c = (-(Mw**2+2*Pt_scalar_product)**2+4*El2*(MET_pt**2))/4
    delta = b**2-4*a*c
    mask = delta < 0
    delta[mask] = 0
    res0 = ((-b-np.sqrt(delta))/(2*a))
    res1 = ((-b+np.sqrt(delta))/(2*a))
    res = np.array([res0, res1])
    res=res[np.abs(res).argsort(axis=0),np.arange(res.shape[1])[None,:]]
    res = list(res)
    res.append(~mask)
    return res





def deltaPhi(phi1, phi2):
    dphi = (phi1 - phi2)
    dphi[dphi > np.pi] = 2*np.pi - dphi[dphi > np.pi]
    dphi[dphi < -np.pi] = -2*np.pi - dphi[dphi < -np.pi]
    return dphi


nu_pz = neutrino_pz(get("Muon_pt[:,0]"), get("Muon_eta[:,0]"), get(
    "Muon_phi[:,0]"), get("MET_pt"), get("MET_phi"))

nu_pz_good_low = nu_pz[0][nu_pz[2]]
nu_pz_good_high = nu_pz[1][nu_pz[2]]
nu_pz_bad = nu_pz[0][~nu_pz[2]]

mu_pt = get("Muon_pt[:,0]")
mu_eta = get("Muon_eta[:,0]")
mu_phi = get("Muon_phi[:,0]")
met_pt = get("MET_pt")
met_phi = get("MET_phi")
mu_pt_good = mu_pt[nu_pz[2]]
mu_eta_good = mu_eta[nu_pz[2]]
mu_phi_good = mu_phi[nu_pz[2]]
met_pt_good = met_pt[nu_pz[2]]
met_phi_good = met_phi[nu_pz[2]]
mu_pt_bad = mu_pt[~nu_pz[2]]
mu_eta_bad = mu_eta[~nu_pz[2]]
mu_phi_bad = mu_phi[~nu_pz[2]]
met_pt_bad = met_pt[~nu_pz[2]]
met_phi_bad = met_phi[~nu_pz[2]]

whereis_nu = signal.arrays("LHEPart_pdgId", library="ak")[
    "LHEPart_pdgId"][:, [4, 7]].to_numpy()
nu_eta_LHE = signal.arrays("LHEPart_eta", library="ak")[
    "LHEPart_eta"][:, [4, 7]].to_numpy()
nu_phi_LHE = signal.arrays("LHEPart_phi", library="ak")[
    "LHEPart_phi"][:, [4, 7]].to_numpy()
nu_pt_LHE = signal.arrays("LHEPart_pt", library="ak")[
    "LHEPart_pt"][:, [4, 7]].to_numpy()
nu_LHEmask = np.bitwise_or(whereis_nu == 14, whereis_nu == -14)

nu_phi_LHE = nu_phi_LHE[nu_LHEmask]
nu_phi_LHE_good = nu_phi_LHE[nu_pz[2]]
nu_phi_LHE_bad = nu_phi_LHE[~nu_pz[2]]

nu_pt_LHE = nu_pt_LHE[nu_LHEmask]
nu_pt_LHE_good = nu_pt_LHE[nu_pz[2]]
nu_pt_LHE_bad = nu_pt_LHE[~nu_pz[2]]

nu_eta_LHE = nu_eta_LHE[nu_LHEmask]
nu_eta_LHE_good = nu_eta_LHE[nu_pz[2]]
nu_eta_LHE_bad = nu_eta_LHE[~nu_pz[2]]


nu_pz_LHE = nu_pt_LHE*np.sinh(nu_eta_LHE)
nu_pz_LHE_good = nu_pz_LHE[nu_pz[2]]
nu_pz_LHE_bad = nu_pz_LHE[~nu_pz[2]]

""" np.save("nu_pz",nu_pz[0])
np.save("nu_pz_det_mask",nu_pz[2]) """


#%% W mass
#You shoul vectorize this function. Maybe there is something else than ROOT.Math.PtEtaPhiMVector
def Wmass(lept_pt, lept_eta, lept_phi, MET_pt, MET_phi, nu_pz, lept_mass=0.105):
    nu_4V = ak.zip(
        {
            "pt": MET_pt,
            "eta": np.arcsinh(nu_pz/MET_pt),
            "phi": MET_phi,
            "mass": np.zeros_like(MET_pt),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    
    mu_4V = ak.zip(
        {
            "pt": lept_pt,
            "eta": lept_eta,
            "phi": lept_phi,
            "mass": np.ones_like(lept_pt)*0.105,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    
    Wmass=ak.to_numpy((nu_4V+mu_4V).mass)
    return Wmass



Wmass_good_low = Wmass(mu_pt_good, mu_eta_good, mu_phi_good,
                       met_pt_good, met_phi_good, nu_pz_good_low)
Wmass_good_high = Wmass(mu_pt_good, mu_eta_good, mu_phi_good,
                        met_pt_good, met_phi_good, nu_pz_good_high)
Wmass_bad = Wmass(mu_pt_bad, mu_eta_bad, mu_phi_bad,
                  met_pt_bad, met_phi_bad, nu_pz_bad)
#%% W mass plot
plt.figure(figsize=(8, 6))
plt.rc('legend', fontsize=13)
h = Histogrammer(bins=50, histrange=(50, 785), xlabel=r"$M_W$ [GeV]",log="y",grid=False)
h.add_hist(Wmass_bad, alpha=1, label="$\Delta=0$ (imposed)",
           color="dodgerblue", edgecolor='blue')
h.add_hist(Wmass_good_low, alpha=0.8, label="$\Delta>0$", color=mcolors.XKCD_COLORS["xkcd:golden yellow"], edgecolor="black")
#plt.xlim(0,1200)
h.plot()
plt.savefig("./images/Wmass_reconstructed.png")

#%% Pz nu
plt.figure(figsize=(8, 6))
plt.rc('legend', fontsize=13)
h=Histogrammer(bins=75, histrange=(0, 2300), xlabel=f"$P_z^{{\\nu}}$ [GeV]",log="y")
h.add_hist(np.abs(nu_pz_good_high),alpha=0.9,
         label=r"MaxAbs $\Delta \geq 0  $",color="dodgerblue", edgecolor='blue', linewidth=1.5)
h.add_hist(np.abs(nu_pz_good_low), alpha=1,
         label=r"MinAbs $\Delta \geq 0$", color=mcolors.XKCD_COLORS["xkcd:golden yellow"],linewidth=1.5, edgecolor="black")
h.add_hist(np.abs(nu_pz_bad), alpha=0.45,
         label=r"$\Delta=0 $ (imposed)", color="fuchsia", edgecolor='violet',  linewidth=1.5)
h.plot()

plt.savefig("./images/Pznu_reconstructed.png")

#%% Pz nu LHE comparison

plt.figure(figsize=(18, 12))
mpl.rc("font", size=18)
plt.rc('legend', fontsize=13)
histrangeabs=(0,300)
histrangediff=(-300,300)
plt.subplot(2, 2, 1)
h1=Histogrammer(bins=75, histrange=histrangeabs,
             xlabel=f"$|P_z^{{\\nu}}|$ [GeV]",  grid=False)
h1.add_hist(np.abs(nu_pz_good_high), alpha=0.8, label=r"MaxAbs $\Delta \geq 0$",
              color="dodgerblue", edgecolor='blue', linewidth=2.5)

h1.add_hist(np.abs(nu_pz_good_low), alpha=0.7, label=r"MinAbs $\Delta \geq 0$",
              color=mcolors.XKCD_COLORS["xkcd:golden yellow"], linewidth=2.5, edgecolor="black")
h1.add_hist(np.abs(nu_pz_LHE_good), alpha=0.7, label=r"LHE",
            edgecolor="fuchsia", color='violet', linewidth=2)

h1.plot()
plt.ylim(0,7000)
plt.subplot(2, 2, 3)
h3 = Histogrammer(bins=75, histrange=histrangediff,
                  xlabel=f"$P_z^{{\\nu}}-P_z^{{LHE}}$ [GeV]",  grid=False)
h3.add_hist(nu_pz_good_high-nu_pz_LHE_good, alpha=0.8, label=r"MaxAbs $\Delta \geq 0$", color="dodgerblue", edgecolor='blue', linewidth=2)
h3.add_hist(nu_pz_good_low-nu_pz_LHE_good, alpha=0.7, label=r"MinAbs $\Delta \geq 0$", color=mcolors.XKCD_COLORS["xkcd:golden yellow"], linewidth=2.5, edgecolor="black")
h3.plot()
plt.ylim(0, 7000)

plt.subplot(2, 2, 2)
h2 = Histogrammer(bins=75, histrange=histrangeabs,
                  xlabel=f"$|P_z^{{\\nu}}|$ [GeV]",  grid=False)
h2.add_hist(np.abs(nu_pz_bad), alpha=0.8, label=r"$\Delta = 0$ (imposed)", color="dodgerblue", edgecolor='blue', linewidth=2)
h2.add_hist(np.abs(nu_pz_LHE_bad), alpha=0.7, label=r"LHE", color=mcolors.XKCD_COLORS["xkcd:golden yellow"], linewidth=2.5, edgecolor="black")
h2.plot()
plt.ylim(0, 7000)

plt.subplot(2, 2, 4)
h4 = Histogrammer(bins=75, histrange=histrangediff,
                  xlabel=f"$P_z^{{\\nu}}-P_z^{{LHE}}$ [GeV]",  grid=False)
h4.add_hist(nu_pz_bad-nu_pz_LHE_bad, alpha=1, label=r"$\Delta = 0 (imposed)$", color="dodgerblue", edgecolor='blue', linewidth=2)
h4.plot()
plt.ylim(0, 3000)
plt.savefig("./images/Pznu_LHE_comparison.png")

#%% MET vs LHE

plt.figure(figsize=(12, 12))
mpl.rc("font", size=18)
plt.rc('legend', fontsize=13)
plt.subplot(2, 2, 1)
h1 = Histogrammer(N=False,bins=75, histrange=(0,1000),
                  xlabel=f"$P_t$ [GeV]", log="y", grid=False)
h1.add_hist(met_pt, alpha=1, label=r"MET", color="dodgerblue", edgecolor='blue', linewidth=2)
h1.add_hist(nu_pt_LHE, alpha=0.7, label=r"LHE", color=mcolors.XKCD_COLORS["xkcd:golden yellow"], linewidth=1, edgecolor="black")
h1.plot()



plt.subplot(2, 2, 3)
h3 = Histogrammer(N=False,bins=75, histrange=(-220,220),
                  xlabel=f"$\Delta P_t$ [GeV]", log="y", grid=False)
h3.add_hist(met_pt-nu_pt_LHE, alpha=1, label=r"MET-LHE", color="dodgerblue", edgecolor='black', linewidth=2)
h3.plot()


plt.subplot(2, 2, 2)
h2 = Histogrammer(N=False,bins=75, histrange=(-3.14,3.14),
                  xlabel=f"$\phi$", grid=False, ylabel="")
h2.add_hist(nu_phi_LHE, alpha=1, label=r"LHE",
            color=mcolors.XKCD_COLORS["xkcd:golden yellow"], linewidth=2.5, edgecolor="black")
h2.add_hist(met_phi, alpha=0.75, label=r"MET", color="dodgerblue", edgecolor='blue', linewidth=2.5)

h2.plot()

plt.subplot(2, 2, 4)
h4 = Histogrammer(N=False,bins=75, histrange=(-3.14, 3.14),
                  xlabel=f"$\\Delta \\phi$", grid=False,ylabel="")
h4.add_hist(deltaPhi(met_phi, nu_phi_LHE), alpha=1, label=r"MET-LHE", color="dodgerblue", edgecolor='black', linewidth=2)
h4.plot()

plt.savefig("./images/MET_LHE_comparison.png")

# %%
