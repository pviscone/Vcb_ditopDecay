import os
import ROOT
import correctionlib
correctionlib.register_pyroot_binding()

ROOT.EnableImplicitMT()
include_path=os.path.join(os.path.dirname(__file__),"vary_utils.h")

ROOT.gInterpreter.ProcessLine(f'#include "{include_path}"')
ROOT.gInterpreter.Declare('auto JES = JERC->at("Summer19UL18_V5_MC_Total_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JER= JERC->at("Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs");')
ROOT.gInterpreter.Declare('auto bTag=btagging->at("deepJet_shape");')

import copy
def loop_redefine(rdf_list,*func_str_lists):
    rdf_list=copy.copy(rdf_list)
    for i in range(len(rdf_list)):
        for func_str in func_str_lists:
            rdf_list[i]=rdf_list[i].Redefine(func_str[0],func_str[1])
    return rdf_list

def vary(rdf_dict):
    res={}

    for dataset in rdf_dict:
        #nominal corrections
        for i in range(len(rdf_dict[dataset])):
            rdf_dict[dataset][i]=(rdf_dict[dataset][i].Define("JetGen_pt","TakeIdx(Jet_pt,GenJet_pt,Jet_genJetIdx)")
                                                .Define("JetGen_mass","TakeIdx(Jet_mass,GenJet_mass,Jet_genJetIdx)")
                                                .Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"nom")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"nom")')
                                                #.Define("Weights",'ROOT::VecOps::Product(evaluate_btag(bTag,"central",Jet_hadronFlavour,{abs(Jet_eta),Jet_pt,Jet_btagDeepFlavB}))')
                                                .Define("Weights","1.")
                            )


        res[dataset]={"nominal":copy.copy(rdf_dict[dataset]),
                        
                    "JESUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                            ,("Jet_mass","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JERUp":loop_redefine(rdf_dict[dataset],("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"up")')
                                                ,("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"up")')
                            ),
                    "JERDown":loop_redefine(rdf_dict[dataset],("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"down")')
                                                ,("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"down")')
                            ),
                    }

    return res



def vary_weights(rdf,syst):
    pass