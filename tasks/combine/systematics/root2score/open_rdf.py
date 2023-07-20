import ROOT
from glob import glob
import numpy as np

ROOT.EnableImplicitMT()

def build_rdf_dict(sample_dict,bunch_size=10):

    rdf_dict = {}
    weight_dict = {"Muons":{},"Electrons":{}}
    for sample in sample_dict:
        path=sample_dict[sample]["path"]
        
        if type(path)==list:
            file=path
        elif path.endswith(".root"):
            file=path
        else:
            file=glob(path+"/*.root")
            
        file_list=[]
        n_bunches=int(np.ceil(len(file)/bunch_size))
        print(f"Number of bunches: {n_bunches}")
        #! For the semileptonic you have to add the efficiencies directly here and not in the json
        
        if sample=="bkg":
            rdf_dict["semiLept"]=[ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]).Filter("Sum(abs(LHEPart_pdgId)==15)==0","NoTaus") for i in range(n_bunches)]
            rdf_dict["semiLeptTau"]=[ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]).Filter("Sum(abs(LHEPart_pdgId)==15)>0","Taus") for i in range(n_bunches)]
            weight_dict["Muons"]["semiLept"]=sample_dict[sample]["Muons_weight"]*0.193
            weight_dict["Electrons"]["semiLept"]=sample_dict[sample]["Electrons_weight"]*0.156
            weight_dict["Muons"]["semiLeptTau"]=sample_dict[sample]["Muons_weight"]*0.0136
            weight_dict["Electrons"]["semiLeptTau"]=sample_dict[sample]["Electrons_weight"]*0.0095
        else:
            rdf_dict[sample] = [ROOT.RDataFrame("Events", file[i*bunch_size:(i+1)*bunch_size]) for i in range(n_bunches)]
            weight_dict["Muons"][sample]=sample_dict[sample]["Muons_weight"]
            weight_dict["Electrons"][sample]=sample_dict[sample]["Electrons_weight"]
            
            
        
    return rdf_dict,weight_dict