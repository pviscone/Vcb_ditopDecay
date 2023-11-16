#%%
import matplotlib.pyplot as plt
import mplhep
import uproot
import numpy as np


combined=uproot.open("./combined/res/likelihood.root")
muon=uproot.open("./muon/res/likelihood.root")
electron=uproot.open("./electron/res/likelihood.root")

#%%
mplhep.style.use(["CMS"])
plt.rcParams['axes.axisbelow'] = True
vcb2=1.681
def plot_like(file,ax,label, **kwargs):
    r=file["limit"]["r"].array()[1:]
    nll=2*file["limit"]["deltaNLL"].array()[1:]
    
    r1=r[r<1][np.argmin(np.abs(nll[r<1]-1))]
    r2=r[r>1][np.argmin(np.abs(nll[r>1]-1))]
    
    r=r*vcb2
    ax.plot(r,nll,linewidth=2,label=label+fr" $  {vcb2:.3f}^{{+{vcb2*(r2-1):.3f}  }}_{{-{vcb2*(1-r1):.3f}}} $",**kwargs)
    ax.grid()
    ax.legend(title="Stat. only",frameon=True,fontsize=18)
    ax.set_xlabel(r"$|V_{cb}|^2 \times 10^3$")
    ax.set_ylabel(r"$-2\Delta\ln\mathcal{L}$")
    
    ax.plot([0.6,2],[1,1,],color="red",linestyle="--")
    ax.set_xlim(0.8*vcb2,1.2*vcb2)
    ax.set_ylim(0,6)
    mplhep.cms.text("Private Work")
    mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)")
    

    
fig,ax=plt.subplots(1,1,figsize=(8,6))

plot_like(muon,ax,label="Muon",color="dodgerblue")
plot_like(electron,ax,label="Electron",color="red")
plot_like(combined,ax,label="Combined",color="black")
plt.savefig("stat_like.png",dpi=800,bbox_inches="tight")
# %%
