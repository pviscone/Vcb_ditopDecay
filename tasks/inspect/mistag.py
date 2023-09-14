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

def np_or(*args):
    out=args[0]
    for i in range(1,len(args)):
        out=np.bitwise_or(out,args[i])
    return out

def delta_phi(phi1,phi2):
    return np.mod(phi1-phi2+np.pi,2*np.pi)-np.pi

def delta_r(eta1,eta2,phi1,phi2):
    return np.sqrt((eta1-eta2)**2+delta_phi(phi1,phi2)**2)



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




mplhep.style.use("CMS")

file_name="2A83E205-B7CE-3744-86A0-5E8B1E807D44.root"

t=uproot.open(file_name)["Events"]


jet_arr=t.arrays(["Jet_btagDeepFlavB",
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


lhe_arr=t.arrays(["LHEPart_pdgId","LHEPart_eta","LHEPart_phi"])
lhe_arr=lhe_arr[:,2:]


jet_arr["ProbC"]=jet_arr["Jet_btagDeepFlavB"]*jet_arr["Jet_btagDeepFlavCvB"]/(1-jet_arr["Jet_btagDeepFlavCvB"])

mu_arr=mu_arr[np_and(mu_arr["Muon_looseId"],
                    mu_arr["Muon_pfIsoId"]>1,)]

mask=ak.num(mu_arr["Muon_pt"])>=1

mu_arr=mu_arr[mask]
jet_arr=jet_arr[mask]
lhe_arr=lhe_arr[mask]

jet_arr=jet_arr[np_and(jet_arr["Jet_jetId"]>0,
                    jet_arr["Jet_puId"]>0,
                    jet_arr["Jet_pt"]>20,
                    np.abs(jet_arr["Jet_eta"])<4.8,
                    delta_r(jet_arr["Jet_eta"][:],mu_arr["Muon_eta"][:,0],jet_arr["Jet_phi"][:],mu_arr["Muon_phi"][:,0])>0.4
                     )]


mask=np_and(ak.num(jet_arr["Jet_pt"])>=4,
            )

jet_arr=jet_arr[mask]
mu_arr=mu_arr[mask]
lhe_arr=lhe_arr[mask]



jet_arr=jet_arr[ak.argsort(jet_arr["Jet_btagDeepFlavB"],axis=1,ascending=False)]
mask=np_and(jet_arr["Jet_btagDeepFlavB"][:,2]>=0.049,
            mu_arr["Muon_pt"][:,0]>=26,
            np.abs(mu_arr["Muon_eta"][:,0])<2.4,
            ak.sort(jet_arr["Jet_btagDeepFlavCvL"][:,3:],axis=1,ascending=False)[:,0]>0.038,
            ak.sort(jet_arr["Jet_btagDeepFlavCvB"][:,3:],axis=1,ascending=False)[:,0]>0.246,
            )


b_sort=jet_arr[:,:3]
c_sort=ak.singletons(
    jet_arr[:,3:][ak.argsort(jet_arr["ProbC"][:,3:],axis=1,ascending=False)
                  ][:,0]
    )

def matching(jet_arr,lhe_arr,n):
    lhe_mask=np_or(
        np.abs(lhe_arr["LHEPart_pdgId"])<6,
        np.abs(lhe_arr["LHEPart_pdgId"])==21)
    
    lhe_arr=lhe_arr[lhe_mask]
    
    jet_phi,lhe_phi=ak.unzip(ak.cartesian([jet_arr["Jet_phi"],
            lhe_arr["LHEPart_phi"]],nested=True))


    jet_eta,lhe_eta=ak.unzip(ak.cartesian([jet_arr["Jet_eta"],
            lhe_arr["LHEPart_eta"]],nested=True))
    
    dr=delta_r(jet_eta,lhe_eta,jet_phi,lhe_phi)
    dr=ak.mask(dr,dr<0.4)
    idx=ak.argmin(ak.mask(dr,dr<0.4),axis=2)
    pdg=(ak.flatten(lhe_arr["LHEPart_pdgId"])[ak.flatten(idx)]).to_numpy().reshape((len(lhe_arr),n))
    
    res=np.ones((len(jet_arr),n))*-10
    res[pdg.mask]=0
    res[np.abs(pdg)==1]=1
    res[np.abs(pdg)==2]=1
    res[np.abs(pdg)==21]=1
    res[np.abs(pdg)==3]=2
    res[np.abs(pdg)==4]=3
    res[np.abs(pdg)==5]=4
    return res





b_sort=jet_arr[:,:3]
c_sort=ak.singletons(
    jet_arr[:,3:][ak.argsort(jet_arr["ProbC"][:,3:],axis=1,ascending=False)
                  ][:,0]
    )
b_sort_match=matching(b_sort,lhe_arr,3)
c_sort_match=matching(c_sort,lhe_arr,1)

#%%


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

ax[0][0]=plot(ax[0][0],b_sort_match[:,0])
ax[0][0].set_ylim(2e-3,2)
ax[0][0].set_ylabel("Fraction")
ax[0][0].text(0.1,0.85,"Leading P(b)",transform=ax[0][0].transAxes,fontsize=20)

ax[0][1]=plot(ax[0][1],b_sort_match[:,1])
ax[0][1].set_yticklabels([])
ax[0][1].set_ylim(2e-3,2)
ax[0][1].text(0.1,0.85,"Second P(b)",transform=ax[0][1].transAxes,fontsize=20)

ax[1][0]=plot(ax[1][0],b_sort_match[:,2])
ax[1][0].set_ylim(2e-2,2)
ax[1][0].text(0.1,0.85,"Third P(b)",transform=ax[1][0].transAxes,fontsize=20)

ax[1][1]=plot(ax[1][1],c_sort_match[:,0])
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
