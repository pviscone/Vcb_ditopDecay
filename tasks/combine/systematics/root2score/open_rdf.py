import ROOT
from glob import glob
import numpy as np

n_thread=os.environ["ROOT_nTHREAD"]
ROOT.EnableImplicitMT(n_thread)

def build_rdf_dict(sample_dict,bunch_size=10):

    rdf_dict = {}
    Xlumi_events = {"Muons":{},"Electrons":{}}
    for sample in sample_dict:
        path=sample_dict[sample]["path"]
        
        if type(path)==list:
            file=path
        elif path.endswith(".root"):
            file=[path]
        else:
            file=glob(path+"/*.root")
            
        n_bunches=int(np.ceil(len(file)/bunch_size))
        #print(f"Number of bunches: {n_bunches}",flush=True)
        
        
        if sample=="bkg":
            rdf_dict["semiLeptMu"]=[ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]).Filter("Sum(abs(LHEPart_pdgId)==13)>0","Muons") for i in range(n_bunches)]
            rdf_dict["semiLeptEle"]=[ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]).Filter("Sum(abs(LHEPart_pdgId)==11)>0","Electrons") for i in range(n_bunches)]
            rdf_dict["semiLeptTau"]=[ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]).Filter("Sum(abs(LHEPart_pdgId)==15)>0","Taus") for i in range(n_bunches)]
            Xlumi_events["semiLeptMu"]=sample_dict[sample]["nEv_lumi"]*0.33
            Xlumi_events["semiLeptEle"]=sample_dict[sample]["nEv_lumi"]*0.33
            Xlumi_events["semiLeptTau"]=sample_dict[sample]["nEv_lumi"]*0.33

        else:
            rdf_dict[sample] = [ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]) for i in range(n_bunches)]
            Xlumi_events[sample]=sample_dict[sample]["nEv_lumi"]
            
            
            
        
    return rdf_dict,Xlumi_events