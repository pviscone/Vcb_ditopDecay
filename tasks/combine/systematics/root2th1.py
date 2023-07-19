from root2score.open_rdf import build_rdf_dict
from root2score.vary import vary
from root2score.skim import systematics_cutloop
from root2score.convert2torch import convert2torch
from root2score.JPAmodel.torchdict2score import torchdict2score
from root2score.th1builder import build_TH1
import torch

cuda=torch.device("cuda:0")
cpu=torch.device("cpu")

samples_json="json/exSamples.json"
bunch=1
device=cpu
outfile="hist.root"

print("\n---------------------Building RDFs---------------------",flush=True)
rdf_dict,weight_dict=build_rdf_dict(samples_json)
rdf_dict,syst_dict=vary(rdf_dict)
print("\n-----------------Starting preselection-----------------",flush=True)
rdf_dict=systematics_cutloop(rdf_dict,syst_dict)
print("\n--------------Conversion to torch datasets-------------",flush=True)
torch_dict=convert2torch(rdf_dict)
print("\n----------------Starting DNN evaluation----------------",flush=True)
score_dict=torchdict2score(torch_dict,bunch=bunch,device=device)
print("\n----------------------Building TH1---------------------",flush=True)
build_TH1(score_dict,weight_dict,outfile)
print("\n--------------------------Done-------------------------",flush=True)
