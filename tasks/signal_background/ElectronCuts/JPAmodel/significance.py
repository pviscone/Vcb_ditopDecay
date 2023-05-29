import numpy as np
import matplotlib.pyplot as plt
import mplhep
import hist

def significance_plot(signal_score,bkg_score,bins=20,score_range=(0,3),
                      ylim=None,log=True,xlabel="NN Score",normalize="lumi",ratio_log=False):
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
        signal_weight=np.ones_like(signal_score)*semileptonic_weight*0.363*8.4e-4/(len(signal_score))
        bkg_weight=np.ones_like(bkg_score)*semileptonic_weight*0.352*(1-8.4e-4)/(len(bkg_score))
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

    sig_err=np.sqrt(signal_hist)*binned_signal_score/(signal_hist+1e-10)
    ax[0].errorbar(bin_centers,
                binned_signal_score,
                sig_err,
                fmt=",",color="black")
    bkg_err=np.sqrt(bkg_hist)*binned_bkg_score/(bkg_hist+1e-10)
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
    else:
        if log == True:
            ax[0].set_ylim(0.101,
                           100*np.max([binned_signal_score,binned_bkg_score]))
        else:
            ax[0].set_ylim(0,1.5*np.max([binned_signal_score,binned_bkg_score]))







    fom=binned_signal_score**2/(binned_bkg_score+binned_signal_score+1e-10)
    Q=(np.sum(fom[~np.isnan(fom)]))

    ratio_err1=sig_err*(binned_signal_score**2+2*binned_signal_score*binned_bkg_score)/(binned_bkg_score+binned_signal_score+1e-10)**2
    ratio_err2=bkg_err/(binned_bkg_score+binned_signal_score+1e-10)**2
    ratio_err=np.sqrt(ratio_err1**2+ratio_err2**2)
    ax[1].errorbar(bin_centers,fom,ratio_err,np.ones_like(fom)*(bin_centers[1]-bin_centers[0])/2,fmt=".",color="black",markersize=5)
    ax[1].set_ylabel("$S^2/(S+B)$",fontsize=22)
    ax[1].set_xlabel(xlabel)

    if ratio_log==True:
        ax[1].set_yscale("log")
        ax[1].set_ylim((np.min(fom[~np.isnan(fom)])-np.min(ratio_err[~np.isnan(ratio_err)]))/5,
                       5*(np.max(fom[~np.isnan(fom)])+np.max(ratio_err[~np.isnan(ratio_err)])))
    else:
        ax[1].set_ylim(-0.1,1.3*(np.max(fom[~np.isnan(fom)])+np.max(ratio_err[~np.isnan(ratio_err)])))
    ax[1].grid(linestyle=":")

    
    mplhep.cms.text("Private Work",ax=ax[0])
    mplhep.cms.lumitext(r"$\mathcal{Q}=$"+f"{Q:.1f}"+", $138 fb^{-1}(13 TeV)$",ax=ax[0],fontsize=22)


    #print("Q=",Q)
    #print(f"Error: {1/np.sqrt(Q)}")
def make_hist(hist_dict,xlim=None,bins=None,log=False,ylim=None,significance=True,**kwargs):
    mplhep.style.use("CMS")
    assert xlim is not None
    assert bins is not None
    categories=list(hist_dict.keys())
    stack_categories=[i for i in hist_dict if hist_dict[i]["stack"]==True]
    
    ax = hist.axis.Regular(bins, xlim[0], xlim[1], flow=False, name="x")
    cax=hist.axis.StrCategory(stack_categories, name="type")
    stack_hist=hist.Hist(ax,cax,storage=hist.storage.Weight())
    
    hist_list=[]
    stack_color=[]
    no_stack_color=[]
    no_stack_label=[]
    no_stack_histtype=[]
    for cat in categories:
        if hist_dict[cat]["stack"] is True:
            stack_hist.fill(hist_dict[cat]["data"],type=cat,weight=hist_dict[cat]["weight"]/len(hist_dict[cat]["data"]))
            stack_color.append(hist_dict[cat]["color"])
        else:
            h=hist.Hist(hist.axis.Regular(bins, xlim[0], xlim[1], flow=False, name="x"),storage=hist.storage.Weight())
            h.fill(hist_dict[cat]["data"],weight=hist_dict[cat]["weight"]/len(hist_dict[cat]["data"]))
            hist_list.append(h)
            no_stack_color.append(hist_dict[cat]["color"])
            no_stack_label.append(cat)
            no_stack_histtype.append(hist_dict[cat]["histtype"])
            
    stack=stack_hist.stack("type")
    tot=sum(stack)
    
    hist_list[0].plot_ratio(tot)
    ax1=plt.subplot(211)
    ax1.clear()
    
    
    linewidth=[0]*len(stack_color)
    linewidth[-1]=1.
    stack.plot(stack=True,histtype="fill",color=stack_color,edgecolor=["black"]*len(stack_categories),linewidth=linewidth)
    mplhep.histplot(tot,color="black",histtype="errorbar",markersize=0,**kwargs)
    for i in range(0,len(no_stack_color)):
        mplhep.histplot(hist_list[i],color=no_stack_color[i],histtype=no_stack_histtype[i],linewidth=1.8,label=no_stack_label[i],yerr=False,xerr=False,**kwargs)
        #plt.hist(hist_dict[no_stack_label[i]]["data"],range=(xlim[0],xlim[1]),bins=bins,weights=np.ones(len(hist_dict[no_stack_label[i]]["data"]))*hist_dict[no_stack_label[i]]["weight"]/len(hist_dict[no_stack_label[i]]["data"]),histtype="step",color=hist_dict[no_stack_label[i]]["color"],hatch="/")

        
    if log is True:
        plt.yscale("log")
        
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    
    plt.grid()
    plt.legend(prop={'size': 14})
    plt.ylabel("Counts")
    mplhep.cms.text("Private Work")
    
    ax2=plt.subplot(212)
    if significance:
        ax2.clear()
        
        sig=sum(hist_list)
        Q=hist.Hist(hist.axis.Regular(bins,xlim[0],xlim[1],flow=False,name="x"))
        
        Q_array=sig.values()**2/(sig.values()+tot.values()+1e-10)
        
        sig_err=[np.sqrt(i) for i in sig.values()]
        bkg_err=[np.sqrt(i) for i in tot.values()]
        sig_val=sig.values()
        bkg_val=tot.values()
        yerr=np.sqrt((((2*sig_val/(sig_val+bkg_val+1e-10))-(sig_val**2/(sig_val+bkg_val+1e-10)**2))*sig_err)**2+((sig_val**2/(sig_val+bkg_val+1e-10)**2)*bkg_err)**2)
        
        
        plt.xlabel("NN score")
        for idx,q in enumerate(Q_array):
            Q[idx]=q
        Q_value=np.sum(Q_array)
        mplhep.cms.lumitext(r"$\mathcal{Q}=$"+f"{Q_value:.1f}"+", $138 fb^{-1}(13 TeV)$",ax=ax1,fontsize=18)
        mplhep.histplot(Q,yerr=yerr,xerr=True,histtype="errorbar",markersize=4,color="black",**kwargs)
        plt.ylabel("$\\frac{S^2}{S+B}$")
    plt.grid()
    if log is True:
        plt.yscale("log")
    return ax1,ax2
    
