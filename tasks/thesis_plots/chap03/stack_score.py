
#%%
import uproot
import numpy as np
import hist
import matplotlib.pyplot as plt
import mplhep
mplhep.style.use("CMS")
plt.rcParams['axes.axisbelow'] = True

file_name="/scratchnvme/pviscone/Vcb_ditopDecay/tasks/combine/systematics/hist.root"
file=uproot.open(file_name)

#%%
channels=["Muons","Electrons"]
procs=[key for key in file["Muons"].keys() if "_" not in key]

def sorted_procs(channel,reverse=False):
    processes={key:np.sum(file[channel][key].counts()) for key in procs}
    # list of key sorted for dict value
    return list(sorted(processes.keys(), key=lambda k: processes[k],reverse=reverse))
    

def stack(channel):
    processes=sorted_procs(channel)
    edges=file[channel][processes[0]].to_numpy()[1]
    h=[]
    for process in processes[1:]:
        h.append(file[channel][process].to_hist())
    stack=hist.Stack(*h)
    return stack


def linedge(channel):
    processes=sorted_procs(channel)
    res=np.zeros_like(file[channel][processes[0]].to_numpy()[0])
    for process in processes:
        res=res+file[channel][process].to_numpy()[0]
    return res,file[channel][processes[0]].to_numpy()[1]

def errors_stack(channel,ax,**kwargs):
    h,edge=linedge(channel)
    ax.stairs(h,edges=edge,linewidth=1.2,color="black")
    centers=(edge[1:]+edge[:-1])/2
    errors=np.zeros_like(h)
    for process in sorted_procs(channel):
        err=file[channel][process].errors()
        errors+=err**2
    errors=np.sqrt(errors)
    ax.errorbar(centers,h,errors,fmt=".",color="black", markersize=0,**kwargs)
    
    
def Q(channel):
    processes=sorted_procs(channel)
    res_signal=np.zeros_like(file[channel][processes[0]].to_numpy()[0])
    res_bkg=np.zeros_like(file[channel][processes[0]].to_numpy()[0])
    for process in processes:
        if "signal" in process:
            res_signal=res_signal+file[channel][process].to_numpy()[0]
        else:
            res_bkg=res_bkg+file[channel][process].to_numpy()[0]
    edges=file[channel][processes[0]].to_numpy()[1]
    centers=(edges[1:]+edges[:-1])/2
    q=res_signal**2/(res_signal+res_bkg)
    return q,edges
    
save=True
#%%
#nice palette of 12 colors
col=["#011993","gold","plum","#5f9b8c","lightcoral","#4376c9","#00cccc","#91d36e","khaki","#f9845f","firebrick","#d9f9f3"]


s=stack("Muons")
#2 subplots
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[3,1]})
s.plot(ax=ax1,stack=True,histtype="fill",color=col[1:])
ax1.set_yscale("log")
ax1.grid()

ax1.set_ylim(1,1e7)
errors_stack("Muons",ax1)
ax1.set_xlabel("")
ax1.set_ylabel("Events")
ax1.legend(fontsize=17,frameon=True,facecolor="white")
mplhep.cms.text("Private Work",ax=ax1)
mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax1)
ax2.stairs(*Q("Muons"),color="red",linewidth=2,)
ax2.set_yscale("log")
ax2.grid()
ax2.set_xlabel("atanh(DNN score)")
ax2.set_title("Muon Channel",loc="left",fontsize=22)
ax2.set_ylabel(r"$\frac{S^2}{S+B}$")
mplhep.cms.lumitext(f"Q={np.sum(Q('Muons')[1]):.2f}",ax=ax2)
if save:
    plt.savefig("stack_score_muons.pdf",bbox_inches='tight')


# %%
colors={sorted_procs("Muons")[i]:col[i] for i in range(len(col))}
del colors["signalMu;1"]
procs_ele=sorted_procs("Electrons")[1:]

col_ele=[colors[proc] for proc in procs_ele]

s=stack("Electrons")
#2 subplots
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[3,1]})
s.plot(ax=ax1,stack=True,histtype="fill",color=col_ele)
ax1.set_yscale("log")
ax1.grid()
ax1.legend(fontsize=17,frameon=True,facecolor="white")
ax1.set_ylim(1,1e7)
#ax1.set_xlim(0,8)
errors_stack("Electrons",ax1)
ax1.set_xlabel("")
ax1.set_ylabel("Events")
mplhep.cms.text("Private Work",ax=ax1)
mplhep.cms.lumitext("138 fb$^{-1}$ (13 TeV)",ax=ax1)
ax2.stairs(*Q("Electrons"),color="red",linewidth=2,)
ax2.set_yscale("log")
ax2.grid()
ax2.set_xlabel("atanh(DNN score)")
ax2.set_ylabel(r"$\frac{S^2}{S+B}$")
ax2.set_title("Electron Channel",loc="left",fontsize=22)
mplhep.cms.lumitext(f"$Q={np.sum(Q('Electrons')[1]):.2f}$",ax=ax2)
if save:
    plt.savefig("stack_score_electrons.pdf",bbox_inches='tight')