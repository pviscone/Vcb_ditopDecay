import ROOT
import numpy as np


bins={}
bins["Muons"]=np.concatenate((np.linspace(0,4.6,40),np.array([6.5])))
bins["Electrons"]=np.concatenate((np.linspace(0,3.85,38),np.array([5.5])))


def build_TH1(score_dict,weight_dict,outfile):
    file=ROOT.TFile(outfile,"RECREATE")
    
    
    for cut in score_dict:
        directory=file.mkdir(cut)
        directory.cd()
        for dataset in score_dict[cut]:
            for syst in score_dict[cut][dataset]:
                data=np.arctanh(score_dict[cut][dataset][syst])
                weight_array=weight_dict[cut][dataset][syst].astype("float")
                
                if syst=="nominal":
                    name=f"{dataset}"
                else:
                    name=f"{dataset}_syst_{syst}"
                hist=ROOT.TH1F(name,f"{cut}_{dataset}_{syst};DNN score;Counts",len(bins[cut])-1,bins[cut])
                hist.Sumw2()
                hist.FillN(len(data),data,weight_array)
                hist.Write()

    file.Close()

if __name__=="__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dict",type=str,help="Path to score dictionary")
    parser.add_argument("--out",type=str,help="output root file")
    args=parser.parse_args()
    
    score_dict=torch.load(args.score_dict)
    build_TH1(score_dict["score_dict"],score_dict["weight_dict"],args.out)
