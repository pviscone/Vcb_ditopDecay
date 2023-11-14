import os
import copy
import awkward as ak
import ROOT
import correctionlib
correctionlib.register_pyroot_binding()

n_thread=int(os.environ["ROOT_nTHREAD"])
ROOT.EnableImplicitMT(n_thread)
include_path=os.path.join(os.path.dirname(__file__),"vary_utils.h")

ROOT.gInterpreter.ProcessLine(f'#include "{include_path}"')

ROOT.gInterpreter.Declare('auto JER= JERC->at("Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs");')
ROOT.gInterpreter.Declare('auto bTag=btagging->at("deepJet_shape");')
ROOT.gInterpreter.Declare('auto cTag=ctagging->at("deepJet_shape");')


ROOT.gInterpreter.Declare('auto JESFlavorQCD = JERC->at("Summer19UL18_V5_MC_Regrouped_FlavorQCD_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESRelativeBal = JERC->at("Summer19UL18_V5_MC_Regrouped_RelativeBal_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESHF = JERC->at("Summer19UL18_V5_MC_Regrouped_HF_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESBBEC1 = JERC->at("Summer19UL18_V5_MC_Regrouped_BBEC1_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESEC2 = JERC->at("Summer19UL18_V5_MC_Regrouped_EC2_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESAbsolute = JERC->at("Summer19UL18_V5_MC_Regrouped_Absolute_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESAbsolute2018 = JERC->at("Summer19UL18_V5_MC_Regrouped_Absolute_2018_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESHF2018 = JERC->at("Summer19UL18_V5_MC_Regrouped_HF_2018_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESEC22018 = JERC->at("Summer19UL18_V5_MC_Regrouped_EC2_2018_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESRelativeSample2018 = JERC->at("Summer19UL18_V5_MC_Regrouped_RelativeSample_2018_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESBBEC12018 = JERC->at("Summer19UL18_V5_MC_Regrouped_BBEC1_2018_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JESTotal = JERC->at("Summer19UL18_V5_MC_Regrouped_Total_AK4PFchs");')


import copy
def loop_redefine(rdf_list,*func_str_lists):
    rdf_list=copy.copy(rdf_list)
    for i in range(len(rdf_list)):
        for func_str in func_str_lists:
            rdf_list[i]=rdf_list[i].Redefine(func_str[0],func_str[1])
    return rdf_list

def vary(rdf_dict,weight_syst_list=[]):
    res={}
    sum_preWeights={}
    for dataset in rdf_dict:
        #nominal corrections
        rdf_list=copy.copy(rdf_dict[dataset])
        for i in range(len(rdf_dict[dataset])):
            rdf_dict[dataset][i]=(rdf_list[i]
                .Define("JetGen_pt","TakeIdx(Jet_pt,GenJet_pt,Jet_genJetIdx)")
                .Define("JetGen_mass","TakeIdx(Jet_mass,GenJet_mass,Jet_genJetIdx)")
                .Define("originalJet_pt","Jet_pt")
                .Define("originalJet_mass","Jet_mass")
                #.Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"nom")')
                #.Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"nom")')
                .Define("GenWeights","genWeight/abs(genWeight)")
                #.Define("JetWeights",'(evaluate_btag(bTag,Jet_hadronFlavour,abs(Jet_eta),Jet_pt,Jet_btagDeepFlavB))')
                #.Redefine("JetWeights",'JetWeights*(evaluate_ctag(cTag,Jet_hadronFlavour,Jet_btagDeepFlavCvL,Jet_btagDeepFlavCvB))')
                .Define("btagW","ROOT::RVec<float>(Jet_pt.size(),1.f)")
                .Define("ctagW","ROOT::RVec<float>(Jet_pt.size(),1.f)")
                )

            
            for syst in weight_syst_list:
                syst_name=""
                if "Up" in syst:
                    syst_name="up_"+syst.split("tag_")[-1].split("Up")[0]
                elif "Down" in syst:
                    syst_name="down_"+syst.split("tag_")[-1].split("Down")[0]
                

                if "btag" in syst:
                    rdf_dict[dataset][i]=rdf_dict[dataset][i].Define(f"btagW_{syst}",f'(vary_btag(bTag,"{syst_name}",Jet_hadronFlavour,abs(Jet_eta),Jet_pt,Jet_btagDeepFlavB,btagW))')
                
                elif "ctag" in syst:
                    rdf_dict[dataset][i]=rdf_dict[dataset][i].Define(f"ctagW_{syst}",f'(vary_ctag(cTag,"{syst_name}",Jet_hadronFlavour,Jet_btagDeepFlavCvL,Jet_btagDeepFlavCvB,ctagW))')

    

                


        res[dataset]={"nominal":copy.copy(rdf_dict[dataset]),
                        
                    "JESFlavorQCDUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESFlavorQCD,{Jet_eta,Jet_pt}))*Jet_pt")
                                            ,("Jet_mass","(1+evaluate(JESFlavorQCD,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESFlavorQCDDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESFlavorQCD,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1-evaluate(JESFlavorQCD,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESRelativeBalUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESRelativeBal,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESRelativeBal,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESRelativeBalDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESRelativeBal,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESRelativeBal,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESHFUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESHF,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESHF,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESHFDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESHF,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESHF,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESBBEC1Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESBBEC1,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESBBEC1,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESBBEC1Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESBBEC1,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESBBEC1,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESEC2Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESEC2,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESEC2,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESEC2Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESEC2,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESEC2,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESAbsoluteUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESAbsolute,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESAbsolute,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESAbsoluteDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESAbsolute,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESAbsolute,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESAbsolute2018Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESAbsolute2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESAbsolute2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESAbsolute2018Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESAbsolute2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESAbsolute2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESHF2018Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESHF2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESHF2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESHF2018Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESHF2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESHF2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESEC22018Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESEC22018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESEC22018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESEC22018Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESEC22018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESEC22018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESRelativeSample2018Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESRelativeSample2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESRelativeSample2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESRelativeSample2018Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESRelativeSample2018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESRelativeSample2018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESBBEC12018Up":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESBBEC12018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESBBEC12018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESBBEC12018Down":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESBBEC12018,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESBBEC12018,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESTotalUp":loop_redefine(rdf_dict[dataset],("Jet_pt","(1+evaluate(JESTotal,{Jet_eta,Jet_pt}))*Jet_pt")
                                                ,("Jet_mass","(1+evaluate(JESTotal,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESTotalDown":loop_redefine(rdf_dict[dataset],("Jet_pt","(1-evaluate(JESTotal,{Jet_eta,Jet_pt}))*Jet_pt")
                                                    ,("Jet_mass","(1-evaluate(JESTotal,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                                        
                    "JERUp":loop_redefine(rdf_dict[dataset],("Jet_pt",'JetGen_pt+(originalJet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"up")/evaluate(JER,{Jet_eta},"nom")')
                                                ,("Jet_mass",'JetGen_mass+(originalJet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"up")/evaluate(JER,{Jet_eta},"nom")')
                            ),
                    "JERDown":loop_redefine(rdf_dict[dataset],("Jet_pt",'JetGen_pt+(originalJet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"down")/evaluate(JER,{Jet_eta},"nom")')
                                                ,("Jet_mass",'JetGen_mass+(originalJet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"down")/evaluate(JER,{Jet_eta},"nom")')
                            ),
                    }

        
    return res



