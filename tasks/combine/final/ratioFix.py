
import uproot
import numpy as np
import matplotlib.pyplot as plt
import os
import ROOT
import statsmodels.api as sm
import mplhep
import argparse
mplhep.style.use("CMS")



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
    



def save_hist(hist_dict,file_out):
    temp=uproot.recreate("temp.root")
    smoothed_ratioUp =hist_dict["smoothedUp"]
    smoothed_ratioDown =hist_dict["smoothedDown"]
    center=hist_dict["center"]

    hUp_smooth=list(hist_dict["histUp"])
    hUp_smooth[0]=center*smoothed_ratioUp
    
    hDown_smooth=list(hist_dict["histDown"])
    hDown_smooth[0]=center*smoothed_ratioDown

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



input_file="hist_smooth.root"
out="fake.root"


f=uproot.open(input_file)
out=uproot.recreate(out)

regions=[key.split(";1")[0] for key in f.keys() if "/" not in key]
samples=[key.split("/")[1].split(";1")[0] for key in f.keys() if (f"{regions[0]}/" in key and "_syst_" not in key)]
systs=[key.split("_syst_")[1].split(";")[0].split("Up")[0].split("Down")[0] for key in f.keys() if (f"{regions[0]}/{samples[0]}_syst_" in key and "Up" in key)]

def cheat(hist_dict,region=None,syst=None):
    assert region is not None
    if "Muons" in region:
        signal_h=compute_ratio(f,region,"signalMu",syst)
    elif "Electrons" in region:
        signal_h=compute_ratio(f,region,"signalEle",syst)
    x=hist_dict["x"]
    hist_dict["smoothedUp"] =signal_h["ratioUp"]
    hist_dict["smoothedDown"] =signal_h["ratioDown"]
    return hist_dict

    
    
#!Smooth everything
systs_to_smooth=[syst for syst in systs if "tag" not in syst ]
print(systs_to_smooth)
for region in regions:
    for sample in samples:
        out[f"{region}/{sample}"]=f[f"{region}/{sample}"]
        for syst in systs:
            if syst in systs_to_smooth:
                h_dict=compute_ratio(f,region,sample,syst)
                h_dict=cheat(h_dict,region=region,syst=syst)
                out=save_hist(h_dict,out)

            else:
                out[f"{region}/{sample}_syst_{syst}Up"]=f[f"{region}/{sample}_syst_{syst}Up"]
                out[f"{region}/{sample}_syst_{syst}Down"]=f[f"{region}/{sample}_syst_{syst}Down"]
                pass

out.close()
    