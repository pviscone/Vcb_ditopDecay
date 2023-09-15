#%%
import uproot
import matplotlib.pyplot as plt
import mplhep

new=uproot.open("new.root")["Muons"]
old=uproot.open("old.root")["Muons"]


#%%
#! new
mplhep.style.use("CMS")
fig, ax=plt.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[3,1],"hspace":0.0})


bkg_exist=False
for old_key in old.keys():
    if old_key!="signal;1":
        if not bkg_exist:
            bkg=old[old_key].to_hist()
            bkg_exist=True
        else:
            bkg=bkg+old[old_key].to_hist()
            
        
            
            
mplhep.histplot(bkg,ax=ax[0],color="dodgerblue",label="old_bkg")
mplhep.histplot(old["signal"],ax=ax[0],color="orange",label="old_sig")

mplhep.histplot(new["semiLeptMu"],ax=ax[0],color="red",ls="--",label="new_bkg")
mplhep.histplot(new["signalMu"],ax=ax[0],color="green",label="new_sig",ls="--")

ax[0].legend()

ax[0].set_yscale("log")
ax[0].grid()
x_old=bkg.axes[0].centers
x_new=new["semiLeptMu"].to_hist().axes[0].centers


ratio_old=old["signal"].values()**2/(bkg.values()+old["signal"].values())

ratio_new=new["signalMu"].values()**2/(new["semiLeptMu"].values()+new["signalMu"].values())

ax[1].plot(x_new,ratio_new,color="red",ls="--",label="new")
ax[1].plot(x_old,ratio_old,color="dodgerblue",label="old")
ax[1].legend()
ax[1].set_yscale("log")
ax[1].grid()

mplhep.cms.lumitext(rf"$Q_{{old}}={sum(ratio_old):.2f} \quad Q_{{new}}={sum(ratio_new):.2f}$",ax=ax[0])