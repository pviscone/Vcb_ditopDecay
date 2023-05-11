import numpy as np
import matplotlib.pyplot as plt
import mplhep


def significance_plot(signal_score,bkg_score,bins=20,score_range=(0,3),
                      ylim=None,log=True,xlabel="NN Score",normalize="lumi",):
    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[3, 1], sharex=True)
    plt.subplots_adjust(hspace=0)
    mplhep.style.use("CMS")
    plt.rc('axes', axisbelow=True)
    semileptonic_weight=(138e3    #Lumi
                        *832      #Cross section
                        *0.44 #Semileptonic BR
                        *0.33     #Muon fraction
                        )
    if normalize=="lumi":
        signal_weight=np.ones_like(signal_score)*semileptonic_weight*0.518*8.4e-4/(len(signal_score))
        bkg_weight=np.ones_like(bkg_score)*semileptonic_weight*0.5*(1-8.4e-4)/(len(bkg_score))
    elif normalize=="equal":
        n_min=np.min([len(signal_score),len(bkg_score)])
        n_max=np.max([len(signal_score),len(bkg_score)])
        signal_weight=np.ones_like(signal_score)*(n_min/n_max)
        bkg_weight=np.ones_like(bkg_score)*(n_min/n_max)
    elif normalize==False:
        signal_weight=np.ones_like(signal_score)
        bkg_weight=np.ones_like(bkg_score)
    signal_hist,bin_edges=np.histogram(((signal_score)),bins=bins,range=score_range)
    bkg_hist,_=np.histogram(((bkg_score)),bins=bins,range=score_range)
    bin_centers=(bin_edges[1:]+bin_edges[:-1])/2



    binned_signal_score=ax[0].hist(((signal_score)),bins=bins,range=score_range,
                                color="dodgerblue",edgecolor="blue",
                                histtype="stepfilled",alpha=0.8,linewidth=2,
                                weights=signal_weight,
                                label="Signal")[0]
    binned_bkg_score=ax[0].hist(((bkg_score)),bins=bins,range=score_range,
                            color="red",edgecolor="red",histtype="step",
                            linewidth=2,label="Background",hatch="//",
                            weights=bkg_weight,
            )[0]

    sig_err=np.sqrt(signal_hist)*binned_signal_score/signal_hist
    ax[0].errorbar(bin_centers,
                binned_signal_score,
                sig_err,
                fmt=",",color="black")
    bkg_err=np.sqrt(bkg_hist)*binned_bkg_score/bkg_hist
    ax[0].errorbar(bin_centers,
                binned_bkg_score,
                bkg_err,fmt=",",color="black")

    if log==True:
        ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid(linestyle=":")
    ax[0].set_ylabel("Normalized events")
    
    if ylim!=None:
        ax[0].set_ylim(ylim[0],ylim[1])

    mplhep.cms.text("Private Work",ax=ax[0])
    mplhep.cms.lumitext("$138 fb^{-1}$ $(13 TeV)$",ax=ax[0])






    fom=binned_signal_score**2/(binned_bkg_score+binned_signal_score)
    Q=(np.sum(fom[~np.isnan(fom)]))

    ratio_err1=sig_err*(binned_signal_score**2+2*binned_signal_score*binned_bkg_score)/(binned_bkg_score+binned_signal_score)**2
    ratio_err2=bkg_err/(binned_bkg_score+binned_signal_score)**2
    ratio_err=np.sqrt(ratio_err1**2+ratio_err2**2)
    ax[1].errorbar(bin_centers,fom,ratio_err,np.ones_like(fom)*(bin_centers[1]-bin_centers[0])/2,fmt=".",color="black",markersize=5)
    ax[1].set_ylabel("$S^2/(S+B)$",fontsize=22)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylim(0,1.2*(np.max(fom[~np.isnan(fom)])+np.max(ratio_err[~np.isnan(ratio_err)])))
    ax[0].text(0.66,0.765,r"$\sum \frac{s^2}{s+b}=$"+f"{Q:.2f}",transform=ax[0].transAxes,fontsize=18)


    print("Q=",Q)
    print(f"Error: {1/np.sqrt(Q)}")
