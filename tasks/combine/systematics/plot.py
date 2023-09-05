#%%
import os
import uproot
import matplotlib.pyplot as plt
import json
import mplhep
import hist
from matplotlib import gridspec
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy import stats



def poisson_interval(
    values,
    variances = None,
    coverage = None,
):
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    if variances is None:
        interval_min = stats.chi2.ppf((1 - coverage) / 2, 2 * values) / 2.0
        interval_min[values == 0.0] = 0.0  # chi2.ppf produces NaN for values=0
        interval_max = stats.chi2.ppf((1 + coverage) / 2, 2 * (values + 1)) / 2.0
    else:
        scale = np.ones_like(values)
        mask = np.isfinite(values) & (values != 0)
        np.divide(variances, values, out=scale, where=mask)
        counts: np.typing.NDArray[Any] = values / scale
        interval_min = scale * stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
        interval_min[values == 0.0] = 0.0  # chi2.ppf produces NaN for values=0
        interval_max = (
            scale * stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
        )
        interval_max[values == 0.0] = np.nan
    return np.stack((interval_min, interval_max))



def ratio_uncertainty(
    num,
    denom,
    uncertainty_type= "poisson",
) :
    with np.errstate(divide="ignore", invalid="ignore"):
        # Nota bene: x/0 = inf, 0/0 = nan
        ratio = num / denom
    if uncertainty_type == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_variance = num * np.power(denom, -2.0)
        ratio_uncert = np.abs(poisson_interval(ratio, ratio_variance) - ratio)
    elif uncertainty_type == "poisson-ratio":
        # Details: see https://github.com/scikit-hep/hist/issues/279
        p_lim = clopper_pearson_interval(num, num + denom)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_lim: np.typing.NDArray[Any] = p_lim / (1 - p_lim)
            ratio_uncert = np.abs(r_lim - ratio)
    elif uncertainty_type == "efficiency":
        ratio_uncert = np.abs(clopper_pearson_interval(num, denom) - ratio)
    else:
        raise TypeError(
            f"'{uncertainty_type}' is an invalid option for uncertainty_type."
        )
    return ratio_uncert


mplhep.style.use("CMS")


regions=["Muons","Electrons"]

samples=json.load(open("json/samples.json","r"))
del samples["bkg"]
samples=list(samples.keys())+["semiLeptEle","semiLeptMu","semiLeptTau"]

systs=["JES",
      "JER",
        "btag_hf",
        "btag_lf",
        "btag_hfstats1",
        "btag_hfstats2",
        "btag_lfstats1",
        "btag_lfstats2",
        "btag_cferr1",
        "btag_cferr2",
        "ctag_Extrap",
        "ctag_Interp",
        "ctag_LHEScaleWeight_muF",
        "ctag_LHEScaleWeight_muR",
        "ctag_PSWeightFSR",
        "ctag_PSWeightISR",
        "ctag_PUWeight",
        "ctag_Stat",
        "ctag_XSec_BRUnc_DYJets_b",
        "ctag_XSec_BRUnc_DYJets_c",
        "ctag_jer",
        "ctag_jesTotal",
        ]

f=uproot.open("hist.root")

def plot(sample,syst,region):
    plt.figure(figsize=(10,10))

    
    
    hUp=root_folder[f"{sample}_syst_{syst}Up"].to_hist()
    hDown=root_folder[f"{sample}_syst_{syst}Down"].to_hist()
    h=root_folder[f"{sample}"].to_hist()
    
    
    
            
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])


    a0 = plt.subplot(gs[0])
    a0.set_title(f"{region} {sample} {syst}")
    mplhep.histplot(hUp,label=syst+"Up",color="red",ax=a0)
    mplhep.histplot(h,label="nominal",color="green",ax=a0)
    mplhep.histplot(hDown,label=syst+"Down" ,color="dodgerblue",ax=a0)

    a0.set_yscale("log")
    a0.set_xlabel(None)
    #a0.set_xticks([])
    a0.legend()

    up_array=hUp.to_numpy()[0]
    down_array=hDown.to_numpy()[0]
    array=h.to_numpy()[0]

    a1 = plt.subplot(gs[1], sharex = a0)
    a1.sharex(a0)
    a0.sharex(a1)
    
    bins_array=(h.to_numpy()[1][1:]+h.to_numpy()[1][:-1])/2
    up_ratio=up_array/array
    down_ratio=down_array/array
    up_ratio_unc=ratio_uncertainty(up_array,array)
    down_ratio_unc=ratio_uncertainty(down_array,array)
    
    #a1.errorbar(bins_array,up_ratio,up_ratio_unc,label=syst+"Up",fmt="--r.",elinewidth=1,markersize=9)
    #a1.errorbar(bins_array+0.03,down_ratio,down_ratio_unc,label=syst+"Down",color="dodgerblue",fmt="--b.",elinewidth=1,markersize=9)
    a1.errorbar(bins_array,up_ratio,label=syst+"Up",fmt="--r.",elinewidth=1,markersize=9)
    a1.errorbar(bins_array+0.03,down_ratio,label=syst+"Down",color="dodgerblue",fmt="--b.",elinewidth=1,markersize=9)
    a1.plot([h.to_numpy()[1][0],h.to_numpy()[1][-1]],[1,1],color="black")
    
    #a1.bar(bins_array,up_array/array,width=1,histtye="step")
    #a1.set_yscale("log")
    
    concat=np.concatenate((up_ratio,down_ratio))
    n=len(concat)
    concat_unc=np.concatenate((up_ratio_unc[1],down_ratio_unc[1]))
    concat_unc=concat_unc[np.isfinite(concat)]
    concat=concat[np.isfinite(concat)]
    



    a1.set_ylim(np.min(concat[concat_unc<0.15])*0.975,np.max(concat[concat_unc<0.15])*1.025)
    a1.set_xlabel("DNN score")
    a1.set_ylabel("Ratio")

    a0.grid()
    a1.grid()

    
    plt.savefig(f"img/{region}/{sample}/{syst}.png")
    plt.close()

def plot_sample(sample,systs,region):
    for syst in systs:
        plot(sample,syst,region)

processes=[]

parallel=True
if parallel:
    import multiprocessing as mp


for region in regions:
    try:
        os.makedirs(f"img/{region}",exist_ok=True)
        sample=""
        root_folder=f[region]
        for sample in samples:
        
                os.makedirs(f"img/{region}/{sample}",exist_ok=True)
                #for syst in systs:
                if parallel:
                    process=mp.Process(target=plot_sample,args=(sample,systs,region))
                    processes.append(process)
                    process.start()
                else:
                    plot_sample(sample,systs,region)
    except:
        print(f"Passing {region}/{sample}")
        pass
        
if parallel:
    for process in processes:
        process.join()