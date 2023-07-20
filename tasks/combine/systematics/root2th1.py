from root2score.open_rdf import build_rdf_dict
from root2score.vary import vary
from root2score.skim import systematics_cutloop
from root2score.convert2torch import convert2torch
from root2score.JPAmodel.torchdict2score import torchdict2score
from root2score.th1builder import build_TH1
import torch
import json

cuda=torch.device("cuda:0")
cpu=torch.device("cpu")

samples_json="json/exSamples.json"
bunch=1
file_bunch_size=10
device=cpu
outfile="hist.root"

sample_dict=json.load(open(samples_json,"r"))

score_dict={}
weight_dict={"Muons":{},"Electrons":{}}

#! There is a bug in ak.from_rdataframe. If the RDataFrame is too bit, it will crash.
#! Create N rdataframes with N/bunch_size files each and concatenate them afterwards.
for sample in sample_dict:
    sample_dict_temp={sample:sample_dict[sample]}
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!{sample}!!!!!!!!!!!!!!!!!!!!!!!!",flush=True)
    #print("\n---------------------Building RDFs---------------------",flush=True)
    rdf_dict,weight_dict_temp=build_rdf_dict(sample_dict_temp,bunch_size=file_bunch_size)
    
    rdf_dict,syst_dict=vary(rdf_dict)
    #print("\n-----------------Starting preselection-----------------",flush=True)
    rdf_dict=systematics_cutloop(rdf_dict,syst_dict)

    #print("\n--------------Conversion to torch datasets-------------",flush=True)
    for idx,syst in enumerate(syst_dict):
        print(f"\n#######################{syst}#########################",flush=True)
        torch_dict=convert2torch(rdf_dict,syst)
        print("\n----------------Starting DNN evaluation----------------",flush=True)
        score_dict_temp=torchdict2score(torch_dict,bunch=bunch,device=device)
        if idx==0:
            score_dict=score_dict_temp
        else:
            if sample=="bkg":
                score_dict["Muons"]["semiLept"][syst]=score_dict_temp["Muons"]["semiLept"][syst]
                score_dict["Electrons"]["semiLept"][syst]=score_dict_temp["Electrons"]["semiLept"][syst]
                score_dict["Muons"]["semiLeptTau"][syst]=score_dict_temp["Muons"]["semiLeptTau"][syst]
                score_dict["Electrons"]["semiLeptTau"][syst]=score_dict_temp["Electrons"]["semiLeptTau"][syst]
            else:
                score_dict["Muons"][sample][syst]=score_dict_temp["Muons"][sample][syst]
                score_dict["Electrons"][sample][syst]=score_dict_temp["Electrons"][sample][syst]
    
    
    if sample=="bkg":
        weight_dict["Muons"]["semiLept"]=weight_dict_temp["Muons"]["semiLept"]
        weight_dict["Electrons"]["semiLept"]=weight_dict_temp["Electrons"]["semiLept"]
        weight_dict["Muons"]["semiLeptTau"]=weight_dict_temp["Muons"]["semiLeptTau"]
        weight_dict["Electrons"]["semiLeptTau"]=weight_dict_temp["Electrons"]["semiLeptTau"]
    else:
        weight_dict["Muons"][sample]=weight_dict_temp["Muons"][sample]
        weight_dict["Electrons"][sample]=weight_dict_temp["Electrons"][sample]

print("\n----------------------Building TH1---------------------",flush=True)
build_TH1(score_dict,weight_dict,outfile)
print("\n--------------------------Done-------------------------",flush=True)
