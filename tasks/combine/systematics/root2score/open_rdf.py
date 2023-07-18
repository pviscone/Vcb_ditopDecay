import ROOT
import json
from glob import glob

ROOT.EnableImplicitMT()

def build_rdf_dict(json_path):
    sample_dict = json.load(open(json_path))
    rdf_dict = {}
    for sample in sample_dict:
        path=sample_dict[sample]
        if path.endswith(".root"):
            file=path
        else:
            file=glob(path+"/*.root")
        rdf_dict[sample] = ROOT.RDataFrame("Events", file)
    return rdf_dict