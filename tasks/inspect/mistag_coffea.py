#%%

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import mplhep
import matplotlib as mpl

def np_and(*args):
    out=args[0]
    for i in range(1,len(args)):
        out=np.bitwise_and(out,args[i])
    return out

def np_or(*args):
    out=args[0]
    for i in range(1,len(args)):
        out=np.bitwise_or(out,args[i])
    return out


def pdg_refactor(array,n):
    array=ak.fill_none(array,0)
    array=np.abs(array[:,:n].to_numpy())
    array[array==21]=1
    array[array==2]=1
    array[array==3]=2
    array[array==4]=3
    array[array==5]=4
    return array

mplhep.style.use("CMS")
file_name="2A83E205-B7CE-3744-86A0-5E8B1E807D44.root"


events = NanoEventsFactory.from_root(
    file_name,
    schemaclass=NanoAODSchema.v6,
).events()



events["LHEPart"]=events.LHEPart[:,2:]
events["LHEPart"]=events.LHEPart[np_or(
                                        np.abs(events.LHEPart.pdgId)<6,
                                        np.abs(events.LHEPart.pdgId)==21)]


events["Jet","ProbC"]=events.Jet.btagDeepFlavB*events.Jet.btagDeepFlavCvB/(1-events.Jet.btagDeepFlavCvB)

events["Muon"]=events.Muon[events.Muon.looseId & (events.Muon.pfIsoId>=1)]

events=events[ak.num(events.Muon)>=1]
events["Muon"]=events.Muon[:,0]

events["Jet"]=events.Jet[np_and(
                                events.Jet.jetId>0,
                                events.Jet.puId>0,
                                events.Jet.pt>20,
                                np.abs(events.Jet.eta)<4.8,
                                events.Jet.delta_r(events.Muon)>0.4)]

events=events[ak.num(events.Jet)>=4]

events["Jet"]=events.Jet[ak.argsort(events.Jet.btagDeepFlavB,axis=1,ascending=False)]

events=events[np_and(
                    events.Jet.btagDeepFlavB[:,2]>=0.049,
                    events.Muon.pt>=26,
                    events.Muon.eta<2.4,
                    ak.sort(events.Jet.btagDeepFlavCvL[:,3:],ascending=False,axis=1)[:,0]>0.038,
                    ak.sort(events.Jet.btagDeepFlavCvB[:,3:],ascending=False,axis=1)[:,0]>0.246,)]



events["cSort"]=events.Jet[:,3:][ak.argsort(events.Jet.ProbC[:,3:],axis=1,ascending=False)]

b_lhe,b_dr=events.Jet[:,:3].nearest(events.LHEPart,return_metric=True)
b_lhe=ak.mask(b_lhe,b_dr<0.4)

c_lhe,c_dr=events.cSort[:,:1].nearest(events.LHEPart,return_metric=True)
c_lhe=ak.mask(c_lhe,c_dr<0.4)


#%%
c_pdg=pdg_refactor(c_lhe.pdgId,1)
b_pdg=pdg_refactor(b_lhe.pdgId,3)

x_ticks_labels = ['0','udg','s','c','b']
fig,ax=plt.subplots(2,2,gridspec_kw={"hspace":0,"wspace":0},figsize=(10,10))
mplhep.cms.text("Private Work",loc=0,ax=ax[0][0])


def plot(ax,array):
    weight=np.ones(len(array))/len(array)
    ax.set_axisbelow(True)
    ax.hist(array,range=(0,5),bins=5,density=True,color="dodgerblue",weights=weight)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels('')
    ax.set_xticks(np.arange(5)+0.5,minor=True)
    ax.set_xticklabels(x_ticks_labels, minor=True)
    ax.set_yscale("log")
    ax.grid(which="both",axis="y")
    ax.grid(which="major",axis="x")
    return ax

ax[0][0]=plot(ax[0][0],b_pdg[:,0])
ax[0][0].set_ylim(2e-3,2)
ax[0][0].set_ylabel("Fraction")
ax[0][0].text(0.1,0.85,"Leading P(b)",transform=ax[0][0].transAxes,fontsize=20)

ax[0][1]=plot(ax[0][1],b_pdg[:,1])
ax[0][1].set_yticklabels([])
ax[0][1].set_ylim(2e-3,2)
ax[0][1].text(0.1,0.85,"Second P(b)",transform=ax[0][1].transAxes,fontsize=20)

ax[1][0]=plot(ax[1][0],b_pdg[:,2])
ax[1][0].set_ylim(2e-2,2)
ax[1][0].text(0.1,0.85,"Third P(b)",transform=ax[1][0].transAxes,fontsize=20)

ax[1][1]=plot(ax[1][1],c_pdg[:,0])
ax[1][1].set_yticklabels([])
ax[1][1].set_ylim(2e-2,2)
ax[1][1].set_xlabel("Jet_partonFlavour")
ax[1][1].text(0.1,0.85,"Leading P(c)",transform=ax[1][1].transAxes,fontsize=20)


mplhep.cms.lumitext(r"$t \bar{t} \to b \bar{b} \mu \nu q \bar{q}$",ax=ax[0][1])
#%%


fig,ax=plt.subplots(figsize=(10,10))
h=ax.hist2d(b_pdg[:,2],c_pdg[:,0],range=((0,5),(0,5)),bins=5,norm=mpl.colors.LogNorm(),density=True,cmap=mpl.cm.YlGnBu_r)
ax.set_xticks(np.arange(5))
ax.set_xticklabels('')
ax.set_xticks(np.arange(5)+0.5,minor=True)
ax.set_xticklabels(x_ticks_labels, minor=True)
ax.set_yticks(np.arange(5))
ax.set_yticklabels('')
ax.set_yticks(np.arange(5)+0.5,minor=True)
ax.set_yticklabels(x_ticks_labels, minor=True)

ax.set_xlabel("Third P(b) PF")
ax.set_ylabel("Leading P(c) PF")

fig.colorbar(h[3],ax=ax,norm=mpl.colors.LogNorm(),ticks=[1e-3,1e-2,1e-1,1,1],cmap=mpl.cm.Spectral)
