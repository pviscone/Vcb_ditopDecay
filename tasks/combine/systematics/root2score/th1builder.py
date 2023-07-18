import ROOT
import numpy as np


bins={}
bins["Muons"]=np.concatenate((np.linspace(0,4.3,43),np.array([4.42,4.55,4.72,6.5])))
bins["Electrons"]=np.linspace(0,5,30)


def build_TH1(score_dict,weight_dict,outfile):
    file=ROOT.TFile(outfile,"RECREATE")
    
    
    for cut in score_dict:
        directory=file.mkdir(cut)
        directory.cd()
        for dataset in score_dict[cut]:
            for syst in score_dict[cut][dataset]:
                data=np.arctanh(score_dict[cut][dataset][syst])
                weight=weight_dict[cut][dataset]
                weight_array=np.ones(len(data),dtype="float")*weight/len(data)
                if syst=="nominal":
                    name=f"hist_{dataset}"
                else:
                    name=f"hist_{dataset}_syst_{syst}"
                hist=ROOT.TH1F(name,f"{cut}_{dataset}_{syst};DNN score;Counts",len(bins[cut])-1,bins[cut])
                hist.Sumw2()
                hist.FillN(len(data),data,weight_array)
                hist.Write()

    file.Close()

