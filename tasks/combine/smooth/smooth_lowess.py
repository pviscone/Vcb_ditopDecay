
#%%

import uproot
import numpy as np
import matplotlib.pyplot as plt
import os
import ROOT
import statsmodels.api as sm
import mplhep
import argparse
mplhep.style.use("CMS")


def symmetrize(hist_dict):
    d=(hist_dict["ratioUp"]-hist_dict["ratioDown"])/2
    
    hist_dict["old_ratioUp"]=hist_dict["ratioUp"]
    hist_dict["old_ratioDown"]=hist_dict["ratioDown"]
    hist_dict["ratioUp"]=1+d
    hist_dict["ratioDown"]=1-d
    
    return hist_dict

def smooth(hist_dict,frac=0.2):
    x=hist_dict["x"]
    Up=hist_dict["ratioUp"]
    Down=hist_dict["ratioDown"]
    hist_dict["smoothedUp"] =sm.nonparametric.lowess(exog=x, endog=Up, frac=frac)[:,1]
    hist_dict["smoothedDown"] =sm.nonparametric.lowess(exog=x, endog=Down, frac=frac)[:,1]
    return hist_dict

def compute_ratio(file,regiom,sample,syst):
    syst=syst.split("Up")[0].split("Down")[0]
    hUp=file[f"{region}/{sample}_syst_{syst}Up"].to_numpy()
    h=file[f"{region}/{sample}"].to_numpy()
    hDown=file[f"{region}/{sample}_syst_{syst}Down"].to_numpy()

    ratioUp=hUp[0]/h[0]
    ratioDown=hDown[0]/h[0]
    mask=np.isfinite(ratioUp)
    x=(h[1][1:]+h[1][:-1])/2
    
    return {"x":x,"center":h[0],"cUp":hUp[0],"cDown":hDown[0],"ratioUp":ratioUp,"ratioDown":ratioDown,"mask":mask,"edges":h[1],"hist":h,"histUp":hUp,"histDown":hDown}
    
def plot_ratio(hist_dict):
    mask=hist_dict["mask"]
    ratioUp=hist_dict["ratioUp"][mask]
    ratioDown=hist_dict["ratioDown"][mask]
    x=hist_dict["x"][mask]
    smoothed_ratioUp =hist_dict["smoothedUp"]
    smoothed_ratioDown =hist_dict["smoothedDown"]

    
    fig, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
    
    add_str=""
    if "old_ratioUp" in hist_dict.keys():
            old_ratioUp=hist_dict["old_ratioUp"][mask]
            old_ratioDown=hist_dict["old_ratioDown"][mask]
            ax.plot(x,old_ratioUp,".",color="goldenrod",label="ratioUp",markersize=10)
            ax.plot(x,old_ratioDown,".",color="dodgerblue",label="ratioDown",markersize=10)
            add_str="_symmetrized"
    

    ax.plot(x, ratioUp,linestyle="--", color="goldenrod", label="ratioUp"+add_str)
    ax.plot(x, smoothed_ratioUp, color="red",label="smoothedUp")
    ax.plot(x, ratioDown,linestyle="--",color="dodgerblue", label="ratioDown"+add_str)
    ax.plot(x, smoothed_ratioDown,color="blue", label="smoothedDown")
    ax.plot([x[0],x[-1]],[1,1,],color="black")
    #plt.ylim(np.min([np.min(ratioDown),np.min(ratioUp)]),np.max([np.max(ratioDown),np.max(ratioUp)]))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24),ncols=3,fontsize=16)
    ax.set_title(f"{region}/{sample}_{syst}")
    ax.grid()
    os.makedirs(f"smooth_img/{region}/{sample}",exist_ok=True)
    fig.savefig(f"smooth_img/{region}/{sample}/{syst}.png")
    plt.close()


def save_hist(hist_dict,file_out):
    temp=uproot.recreate("temp.root")
    smoothed_ratioUp =hist_dict["smoothedUp"]
    smoothed_ratioDown =hist_dict["smoothedDown"]
    center=hist_dict["center"]
    mask=hist_dict["mask"]

    hUp_smooth=list(hist_dict["histUp"])
    hUp_smooth[0][mask]=center[mask]*smoothed_ratioUp
    
    hDown_smooth=list(hist_dict["histDown"])
    hDown_smooth[0][mask]=center[mask]*smoothed_ratioDown

    hDown_smooth[0]=hDown_smooth[0].astype("float32")
    hUp_smooth[0]=hUp_smooth[0].astype("float32")
    hDown_smooth[1]=hDown_smooth[1].astype("float32")
    hUp_smooth[1]=hUp_smooth[1].astype("float32")
    temp[f"{region}/{sample}_syst_{syst}Up"]=(hUp_smooth)
    temp[f"{region}/{sample}_syst_{syst}Down"]=(hDown_smooth)

    up_root=temp[f"{region}/{sample}_syst_{syst}Up"].to_pyroot()
    down_root=temp[f"{region}/{sample}_syst_{syst}Down"].to_pyroot()
    
    up_rootF=ROOT.TH1F()
    down_rootF=ROOT.TH1F()
    up_root.Copy(up_rootF)
    down_root.Copy(down_rootF)
    file_out[f"{region}/{sample}_syst_{syst}Up"]=up_rootF
    file_out[f"{region}/{sample}_syst_{syst}Down"]=down_rootF
    os.remove("temp.root")
    return file_out

    
systs_to_smooth=[
        "btag_hf",
        "JES",
        "JER"]



input_file="../systematics/hist.root"
out="hist_smooth.root"
frac=0.4
symm=True
plot=True



f=uproot.open(input_file)
out=uproot.recreate(out)

regions=[key.split(";1")[0] for key in f.keys() if "/" not in key]
samples=[key.split("/")[1].split(";1")[0] for key in f.keys() if (f"{regions[0]}/" in key and "_syst_" not in key)]
systs=[key.split("_syst_")[1].split(";")[0].split("Up")[0].split("Down")[0] for key in f.keys() if (f"{regions[0]}/{samples[0]}_syst_" in key and "Up" in key)]


#!Smooth everything
systs_to_smooth=systs

for region in regions:
    for sample in samples:
        out[f"{region}/{sample}"]=f[f"{region}/{sample}"]
        for syst in systs:
            if syst in systs_to_smooth:
                h_dict=compute_ratio(f,region,sample,syst)
                if symm:
                    h_dict=symmetrize(h_dict)
                h_dict=smooth(h_dict,frac=frac)
                if plot:
                    plot_ratio(h_dict)
                out=save_hist(h_dict,out)

            else:
                out[f"{region}/{sample}_syst_{syst}Up"]=f[f"{region}/{sample}_syst_{syst}Up"]
                out[f"{region}/{sample}_syst_{syst}Down"]=f[f"{region}/{sample}_syst_{syst}Down"]
                pass

out.close()
    
# %%
