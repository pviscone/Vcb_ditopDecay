import ROOT
import json
from glob import glob

ROOT.EnableImplicitMT()

def build_rdf_dict(json_path):
    sample_dict = json.load(open(json_path))
    rdf_dict = {}
    weight_dict = {"Muons":{},"Electrons":{}}
    for sample in sample_dict:
        
        
        
        path=sample_dict[sample]["path"]
        if path.endswith(".root"):
            file=path
        else:
            file=glob(path+"/*.root")
            
            
        #! For the semileptonic you have to add the efficiencies directly here and not in the json
        if sample=="bkg":
            rdf_dict["semiLept"]=ROOT.RDataFrame("Events", file).Filter("Sum(abs(LHEPart_pdgId)==15)==0","NoTaus")
            rdf_dict["semiLeptTau"]=ROOT.RDataFrame("Events", file).Filter("Sum(abs(LHEPart_pdgId)==15)>0","Taus")
            weight_dict["Muons"]["semiLept"]=sample_dict[sample]["Muons_weight"]*0.193
            weight_dict["Electrons"]["semiLept"]=sample_dict[sample]["Electrons_weight"]*0.156
            weight_dict["Muons"]["semiLeptTau"]=sample_dict[sample]["Muons_weight"]*0.0136
            weight_dict["Electrons"]["semiLeptTau"]=sample_dict[sample]["Electrons_weight"]*0.0095
        else:
            rdf_dict[sample] = ROOT.RDataFrame("Events", file)
            weight_dict["Muons"][sample]=sample_dict[sample]["Muons_weight"]
            weight_dict["Electrons"][sample]=sample_dict[sample]["Electrons_weight"]
            
            
        
    return rdf_dict,weight_dict