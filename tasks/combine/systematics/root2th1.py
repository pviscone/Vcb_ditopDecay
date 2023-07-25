from root2score.open_rdf import build_rdf_dict
from root2score.vary import vary
from root2score.skim import Cut
from root2score.rdf2torch.rdf2torch import rdf2torch
from root2score.JPAmodel.torchdict2score import predict, create_model
from root2score.th1builder import build_TH1
from root2score.build_datacard import build_datacard
import torch
import copy
import json
import gc

cuda=torch.device("cuda:0")
cpu=torch.device("cpu")

samples_json="json/samples.json"
bunch=1
file_bunch_size=10
device=cpu
outfile="hist.root"

model={"Muons":create_model("root2score/JPAmodel/state_dict_Muons.pt",device=device),
        "Electrons":create_model("root2score/JPAmodel/state_dict_Electrons.pt",device=device)}

sample_dict=json.load(open(samples_json,"r"))
score_dict={"Muons":{},"Electrons":{}}
weight_dict={"Muons":{},"Electrons":{}}

#! There is a bug in ak.from_rdataframe. If the RDataFrame is too bit, it will crash.
#! Create N rdataframes with N/bunch_size files each and concatenate them afterwards.



def list2updown(syst_list):
    res=[]
    for syst in syst_list:
        if syst!="nominal":
            res.append(syst+"Up")
            res.append(syst+"Down")
    return res



weight_syst_list=[
           "btag_hf",
           "btag_lf",
           "btag_hfstats1",
           "btag_hfstats2",
           "btag_lfstats1",
           "btag_lfstats2",
           "btag_cferr1",
           "btag_cferr2",
           "ctag_Extrap",
           "ctag_Interp",
           "ctag_LHEScaleWeight_muF",
           "ctag_LHEScaleWeight_muR",
           "ctag_PSWeightFSR",
           "ctag_PSWeightISR",
           "ctag_PUWeight",
           "ctag_Stat",
           "ctag_XSec_BRUnc_DYJets_b",
           "ctag_XSec_BRUnc_DYJets_c",
           "ctag_jer",
           "ctag_jesTotal",
           ]

var_syst_list=["JES",
              "JER"]

var_syst_list=list2updown(var_syst_list)
weight_syst_list=list2updown(weight_syst_list)

#!nominal must be the first
syst_list=["nominal"]+var_syst_list

#rdf_dict = {sample:rdf_list}, weight_dict = {cut:{sample:weight}}
rdf_dict,weight_dict_temp=build_rdf_dict(sample_dict,bunch_size=file_bunch_size)

#rdf_dict {sample:{syst:rdf_list}},sum_nominal_weights_dict {sample:sum_nominal_weights(before selection)}
rdf_dict,sum_nominal_weights_dict=vary(rdf_dict,weight_syst_list=weight_syst_list)

#! print also the weight efficiency
for cut in ["Muons","Electrons"]:
    print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Cut: {cut} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",flush=True)
    for sample in rdf_dict:
        print(f"\n##################################### {sample} ######################################",flush=True)
        print(f"\n Number of bunches: {len(rdf_dict[sample]['nominal'])}")
        score_dict[cut][sample]={}
        weight_dict[cut][sample]={}
        

        for syst in syst_list:
            print(f"\n------------------------------------- {syst} -------------------------------------",flush=True)

            rdf=Cut(rdf_dict,sample,syst,cut)
            if syst=="nominal":
                torch_dataset,weight_syst_dict=rdf2torch(rdf,cut=cut,weight_syst_list=weight_syst_list,sum_of_preselection_weights=sum_nominal_weights_dict[sample])
                weight_dict[cut][sample]["nominal"]=weight_syst_dict["nominal"]*weight_dict_temp[cut][sample]
                for weight_syst in weight_syst_list:
                    weight_dict[cut][sample][weight_syst]=weight_syst_dict[weight_syst]*weight_dict_temp[cut][sample]
            else:
                torch_dataset,weight_arr=rdf2torch(rdf,cut=cut,sum_of_preselection_weights=sum_nominal_weights_dict[sample])
                weight_dict[cut][sample][syst]=weight_arr*weight_dict_temp[cut][sample]
            

            del rdf
            gc.collect()
                
            
            print("\nStarting DNN evaluation...",flush=True)
            score_dict[cut][sample][syst]=predict(model[cut],torch_dataset,bunch=bunch)
            if syst=="nominal":
                for weight_syst in weight_syst_list:
                    score_dict[cut][sample][weight_syst]=score_dict[cut][sample]["nominal"]
            
            
            del torch_dataset
            gc.collect()
            


print("\n----------------------Building TH1---------------------",flush=True)
build_TH1(score_dict,weight_dict,outfile)
torch.save({"score_dict":score_dict,"weight_dict":weight_dict},"score_dict.pt")
build_datacard(rdf_dict,syst_list=syst_list+weight_syst_list,autoMCStats=True)
print("\n--------------------------Done-------------------------",flush=True)







