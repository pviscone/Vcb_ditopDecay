
#%%
import uproot
import numpy as np
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
import os
import ROOT

def smooth_plot(file,region,sample,syst,s=3,symmetrize=False):
    syst=syst.split("Up")[0].split("Down")[0]
    hUp=file[f"{region}/{sample}_syst_{syst}Up"].to_numpy()
    h=file[f"{region}/{sample}"].to_numpy()
    hDown=file[f"{region}/{sample}_syst_{syst}Down"].to_numpy()
        

    
    ratioUp=hUp[0]/h[0]
    ratioDown=hDown[0]/h[0]
    mask=np.isfinite(ratioUp)
    x=(h[1][1:]+h[1][:-1])/2
    
    ratioUp=ratioUp[mask]
    ratioDown=ratioDown[mask]
    x=x[mask]
    plt.figure()
    
    add_str=""
    if symmetrize:
        plt.plot(x,ratioUp,".",color="goldenrod",label="ratioUp")
        plt.plot(x,ratioDown,".",color="dodgerblue",label="ratioDown")
        add_str="_symmetrized"
        ratioUp,ratioDown=(2-ratioDown+ratioUp)/2,(2-ratioUp+ratioDown)/2
        
    

    smoothed_ratioUp = splrep(x, ratioUp, s=s)
    smoothed_ratioDown = splrep(x, ratioDown, s=s)
    
    
    
    plt.plot(x, ratioUp,linestyle="--", color="goldenrod", label="ratioUp"+add_str)
    plt.plot(x, BSpline(*smoothed_ratioUp)(x), color="red",label="smoothedUp")
    plt.plot(x, ratioDown,linestyle="--",color="dodgerblue", label="ratioDown"+add_str)
    plt.plot(x, BSpline(*smoothed_ratioDown)(x),color="blue", label="smoothedDown")
    plt.plot([x[0],x[-1]],[1,1,],color="black")
    #plt.ylim(np.min([np.min(ratioDown),np.min(ratioUp)]),np.max([np.max(ratioDown),np.max(ratioUp)]))
    plt.legend()
    plt.title(f"{region}/{sample}_{syst}")
    os.makedirs(f"smooth_img/{region}/{sample}",exist_ok=True)
    plt.savefig(f"smooth_img/{region}/{sample}/{syst}.png")
    plt.close()
    
def smooth_hist(file_in,file_out,region,sample,syst,s=5,symmetrize=False):
    syst=syst.split("Up")[0].split("Down")[0]
    temp=uproot.recreate("temp.root")
    
    hUp=list(file_in[f"{region}/{sample}_syst_{syst}Up"].to_numpy())
    h=list(file_in[f"{region}/{sample}"].to_numpy())
    hDown=list(file_in[f"{region}/{sample}_syst_{syst}Down"].to_numpy())
    

        
    x=(h[1][1:]+h[1][:-1])/2
    ratioUp=hUp[0]/h[0]
    ratioDown=hDown[0]/h[0]
    
        
    if symmetrize:
        ratioUp,ratioDown=(2-ratioDown+ratioUp)/2,(2-ratioUp+ratioDown)/2
    
    
    mask=np.isfinite(ratioUp)
    smoothed_ratioUp = splrep(x[mask], ratioUp[mask], s=s)
    smoothed_ratioDown = splrep(x[mask], ratioDown[mask], s=s)
    
    x=(h[1][1:]+h[1][:-1])/2
    


    hUp_smooted=list(hUp)

    hUp_smooted[0]=(BSpline(*smoothed_ratioUp)(x))*h[0]
    
    hDown_smooted=list(hDown)
    hDown_smooted[0]=(BSpline(*smoothed_ratioDown)(x))*h[0]
    
    hDown_smooted[0]=hDown_smooted[0].astype("float32")
    hUp_smooted[0]=hUp_smooted[0].astype("float32")
    hDown_smooted[1]=hDown_smooted[1].astype("float32")
    hUp_smooted[1]=hUp_smooted[1].astype("float32")
    temp[f"{region}/{sample}_syst_{syst}Up"]=(hUp_smooted)
    temp[f"{region}/{sample}_syst_{syst}Down"]=(hDown_smooted)
    
    
    up_root=temp[f"{region}/{sample}_syst_{syst}Up"].to_pyroot()
    down_root=temp[f"{region}/{sample}_syst_{syst}Down"].to_pyroot()
    
    up_rootF=ROOT.TH1F()
    down_rootF=ROOT.TH1F()
    up_root.Copy(up_rootF)
    down_root.Copy(down_rootF)
    file_out[f"{region}/{sample}_syst_{syst}Up"]=up_rootF
    file_out[f"{region}/{sample}_syst_{syst}Down"]=down_rootF
    
    return file_out
    
    
    
    
systs_to_smooth=["btag_hf",
      "JES",
      "JER"]

f=uproot.open("hist.root")
out=uproot.recreate("smooth.root")

regions=[key.split(";1")[0] for key in f.keys() if "/" not in key]
samples=[key.split("/")[1].split(";1")[0] for key in f.keys() if (f"{regions[0]}/" in key and "_syst_" not in key)]
systs=[key.split("_syst_")[1].split(";")[0].split("Up")[0].split("Down")[0] for key in f.keys() if (f"{regions[0]}/{samples[0]}_syst_" in key and "Up" in key)]

for region in regions:
    for sample in samples:
        out[f"{region}/{sample}"]=f[f"{region}/{sample}"]
        for syst in systs:
            if syst in systs_to_smooth:
                #print(f"{region}/{sample}_{syst}")
                out=smooth_hist(f,out,region,sample,syst,s=5,symmetrize=True)
                #smooth_plot(f,region,sample,syst,s=5,symmetrize=True)
            else:
                out[f"{region}/{sample}_syst_{syst}Up"]=f[f"{region}/{sample}_syst_{syst}Up"]
                out[f"{region}/{sample}_syst_{syst}Down"]=f[f"{region}/{sample}_syst_{syst}Down"]
                #pass

out.close()
os.remove("temp.root")
# %%
