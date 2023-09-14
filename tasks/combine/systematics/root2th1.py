#%%
from root2score.open_rdf import build_rdf_dict
from root2score.vary import vary
from root2score.skim import Cut
from root2score.rdf2torch.rdf2torch import rdf2torch
from root2score.JPAmodel.torchdict2score import predict, create_model
from root2score.th1builder import build_TH1
from root2score.build_datacard import build_datacard
from root2score.utils import list2updown
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
#! There is a bug in ak.from_rdataframe. If the RDataFrame is too bit, it will crash.
#! Create N rdataframes with N/bunch_size files each and concatenate them afterwards.

regions=["Muons",
         "Electrons"
        ]

weight_syst_list=[
           "btag_hf",
           "btag_lf",
           "btag_hfstats1",
           "btag_hfstats2",
           "btag_lfstats1",
           "btag_lfstats2",
           "btag_cferr1",
           "btag_cferr2",
            ]    
'''
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
'''     

var_syst_list=["JES",
              "JER"]

var_syst_list=list2updown(var_syst_list)
weight_syst_list=list2updown(weight_syst_list)

#!nominal must be the first
syst_list=["nominal"]+var_syst_list

#rdf_dict = {sample:rdf_list}, weight_dict = {sample:weight}
rdf_dict,nEv_lumi_dict=build_rdf_dict(sample_dict,bunch_size=file_bunch_size)

#rdf_dict {sample:{syst:rdf_list}},sum_nominal_weights_dict {sample:sum_nominal_weights(before selection)}
rdf_dict=vary(rdf_dict,weight_syst_list=weight_syst_list)

score_dict={}
weight_dict={}
eff_dict={}
for region in regions:
    score_dict[region]={}
    weight_dict[region]={}
    eff_dict[region]={}
    print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Region: {region} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",flush=True)
    for dataset in rdf_dict:
        score_dict[region][dataset]={}
        eff_dict[region][dataset]={}
        print(f"\n##################################### {dataset} ######################################",flush=True)
        print(f"\n Number of bunches: {len(rdf_dict[dataset]['nominal'])}")
        for syst in syst_list:
            print(f"\n------------------------------------- {syst} -------------------------------------",flush=True)
            rdf,eff_dict=Cut(rdf_dict,
                    region=region,
                    dataset=dataset,
                    syst=syst,
                    weight_syst_list=weight_syst_list,
                    eff_dict=eff_dict)

            if syst=="nominal":
                torch_dataset,weight_dict[region][dataset]=rdf2torch(rdf,
                                        region=region,
                                        weight_syst_list=weight_syst_list,
                                        eff=eff_dict[region][dataset][syst],
                                        nEv_lumi=nEv_lumi_dict[dataset])
            else:
                torch_dataset,weight_dict[region][dataset][syst]=rdf2torch(rdf,
                                        region=region,
                                        eff=eff_dict[region][dataset][syst],
                                        nEv_lumi=nEv_lumi_dict[dataset])

            del rdf
            gc.collect()
            
            print("\nStarting DNN evaluation...",flush=True)
            score_dict[region][dataset][syst]=predict(model[region],
                                                     torch_dataset,
                                                     bunch=bunch)
            if syst=="nominal":
                for weight_syst in weight_syst_list:
                    score_dict[region][dataset][weight_syst]=score_dict[region][dataset]["nominal"]
                
            del torch_dataset
            gc.collect()


print("\n----------------------Building TH1---------------------",flush=True)
build_TH1(score_dict,weight_dict,outfile)
torch.save({"score_dict":score_dict,"weight_dict":weight_dict},"score_dict.pt")
build_datacard(rdf_dict,
               regions=regions,
               syst_list=syst_list+weight_syst_list,
               autoMCStats=True)
print("\n--------------------------Done-------------------------",flush=True)

