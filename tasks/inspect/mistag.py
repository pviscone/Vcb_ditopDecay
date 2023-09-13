#%%
import uproot
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

def delta_phi(phi1,phi2):
    return np.mod(phi1-phi2+np.pi,2*np.pi)-np.pi

def delta_r(eta1,eta2,phi1,phi2):
    return np.sqrt((eta1-eta2)**2+delta_phi(phi1,phi2)**2)


mplhep.style.use("CMS")

file_name="2A83E205-B7CE-3744-86A0-5E8B1E807D44.root"

t=uproot.open(file_name)["Events"]


arrays=t.arrays(["Jet_btagDeepFlavB",
                 "Jet_btagDeepFlavCvB",
                 "Jet_btagDeepFlavCvL",
                 "Jet_partonFlavour",
                 "Jet_jetId",
                 "Jet_puId",
                 "Jet_pt",
                 "Jet_eta",
                 "Jet_phi",
                 ])

mu_arr=t.arrays(["Muon_pt",
                 "Muon_eta",
                 "Muon_phi",
                 "Muon_looseId",
                 "Muon_pfIsoId"])

arrays["ProbC"]=arrays["Jet_btagDeepFlavB"]*arrays["Jet_btagDeepFlavCvB"]/(1-arrays["Jet_btagDeepFlavCvB"])

mu_arr=mu_arr[np_and(mu_arr["Muon_looseId"],
                    mu_arr["Muon_pfIsoId"]>1,)]

mask=ak.num(mu_arr["Muon_pt"])>=1

mu_arr=mu_arr[mask]
arrays=arrays[mask]

arrays=arrays[np_and(arrays["Jet_jetId"]>0,
                    arrays["Jet_puId"]>0,
                    arrays["Jet_pt"]>20,
                    np.abs(arrays["Jet_eta"])<4.8,
                    delta_r(arrays["Jet_eta"][:],mu_arr["Muon_eta"][:,0],arrays["Jet_phi"][:],mu_arr["Muon_phi"][:,0])>0.4
                     )]


mask=np_and(ak.num(arrays["Jet_pt"])>=4,
            )

arrays=arrays[mask]
mu_arr=mu_arr[mask]

arrays=arrays[ak.argsort(arrays["Jet_btagDeepFlavB"],axis=1,ascending=False)]
mask=np_and(arrays["Jet_btagDeepFlavB"][:,2]>=0.049,
            mu_arr["Muon_pt"][:,0]>=26,
            np.abs(mu_arr["Muon_eta"][:,0])<2.4,
            ak.sort(arrays["Jet_btagDeepFlavCvL"][:,3:],axis=1,ascending=False)[:,0]>0.038,
            ak.sort(arrays["Jet_btagDeepFlavCvB"][:,3:],axis=1,ascending=False)[:,0]>0.246,
            )

#%%


b_sort=arrays
new_arr=b_sort[:,3:]
c_sort=new_arr[ak.argsort(new_arr["ProbC"],axis=1,ascending=False)]




def pdg_refactor(array,n):
    array=ak.pad_none(array,n+1,axis=1)
    array=ak.fill_none(array,-10)
    array=np.abs(array[:,n].to_numpy())
    array[array==21]=1
    array[array==2]=1
    array[array==3]=2
    array[array==4]=3
    array[array==5]=4
    return array


#define xkcd yellow



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


ax[0][0]=plot(ax[0][0],pdg_refactor(b_sort["Jet_partonFlavour"],0))
ax[0][0].set_ylim(2e-3,2)
ax[0][0].set_ylabel("Fraction")
ax[0][0].text(0.1,0.85,"Leading P(b)",transform=ax[0][0].transAxes,fontsize=20)

ax[0][1]=plot(ax[0][1],pdg_refactor(b_sort["Jet_partonFlavour"],1))
ax[0][1].set_yticklabels([])
ax[0][1].set_ylim(2e-3,2)
ax[0][1].text(0.1,0.85,"Second P(b)",transform=ax[0][1].transAxes,fontsize=20)

ax[1][0]=plot(ax[1][0],pdg_refactor(b_sort["Jet_partonFlavour"],2))
ax[1][0].set_ylim(2e-2,2)
ax[1][0].text(0.1,0.85,"Third P(b)",transform=ax[1][0].transAxes,fontsize=20)




ax[1][1]=plot(ax[1][1],pdg_refactor(c_sort["Jet_partonFlavour"],0))
ax[1][1].set_yticklabels([])
ax[1][1].set_ylim(2e-2,2)
ax[1][1].set_xlabel("Jet_partonFlavour")
ax[1][1].text(0.1,0.85,"Leading P(c)",transform=ax[1][1].transAxes,fontsize=20)


mplhep.cms.lumitext(r"$t \bar{t} \to b \bar{b} \mu \nu q \bar{q}$",ax=ax[0][1])
#%%

fig,ax=plt.subplots(figsize=(10,10))
h=ax.hist2d(pdg_refactor(b_sort["Jet_partonFlavour"],2),pdg_refactor(c_sort["Jet_partonFlavour"],0),range=((0,5),(0,5)),bins=5,norm=mpl.colors.LogNorm(),density=True,cmap=mpl.cm.YlGnBu)
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
