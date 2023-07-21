from root2score.open_rdf import build_rdf_dict
from root2score.vary import vary, vary_weights
from root2score.skim import Cut
from root2score.rdf2torch.rdf2torch import rdf2torch
from root2score.JPAmodel.torchdict2score import predict, create_model
from root2score.th1builder import build_TH1
import torch
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

#!nominal must be the first
syst_list=["nominal",
           "JESUp",
           "JESDown",
           "JERUp",
           "JERDown",]


#rdf_dict = {sample:rdf_list}, weight_dict = {cut:{sample:weight}}
rdf_dict,weight_dict_temp=build_rdf_dict(sample_dict,bunch_size=file_bunch_size)

#rdf_dict {sample:{syst:rdf_list}}
rdf_dict=vary(rdf_dict)

for cut in ["Muons","Electrons"]:
    print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Cut: {cut} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",flush=True)
    for sample in rdf_dict:
        print(f"\n##################################### {sample} ######################################",flush=True)
        print(f"\n Number of bunches: {len(rdf_dict[sample]['nominal'])}")
        score_dict[cut][sample]={}
        weight_dict[cut][sample]={}
        
        #Just to make vscode shutup
        #rdf_nominal=0
        for syst in syst_list:
            print(f"\n------------------------------------- {syst} -------------------------------------",flush=True)
            if "tag" in syst:
                #Add copy.copy to vary weight
                rdf=vary_weights(rdf_nominal,syst)
                pass
            else:
                rdf=Cut(rdf_dict,sample,syst,cut)
                if syst=="nominal":
                    rdf_nominal=rdf

            torch_dataset,weight_arr=rdf2torch(rdf,cut=cut)
            
            if syst!="nominal":
                del rdf
                gc.collect()
                
            
            print("\nStarting DNN evaluation...",flush=True)
            score_dict[cut][sample][syst]=predict(model[cut],torch_dataset,bunch=bunch)
            weight_dict[cut][sample][syst]=weight_arr*weight_dict_temp[cut][sample]
            
            del torch_dataset
            gc.collect()
            
        del rdf_nominal
        gc.collect()


print("\n----------------------Building TH1---------------------",flush=True)
build_TH1(score_dict,weight_dict,outfile)
torch.save({"score_dict":score_dict,"weight_dict":weight_dict},"score_dict.pt")
print("\n--------------------------Done-------------------------",flush=True)







