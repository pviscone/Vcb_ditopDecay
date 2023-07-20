import os
import ROOT
import correctionlib
correctionlib.register_pyroot_binding()

ROOT.EnableImplicitMT()
include_path=os.path.join(os.path.dirname(__file__),"vary_utils.h")

ROOT.gInterpreter.ProcessLine(f'#include "{include_path}"')
ROOT.gInterpreter.Declare('auto JES = JERC->at("Summer19UL18_V5_MC_Total_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JER= JERC->at("Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs");')

def vary(rdf_dict):
    res={}

    for dataset in rdf_dict:
        #nominal corrections
        rdf_dict[dataset]=(rdf_dict[dataset].Define("JetGen_pt","TakeIdx(Jet_pt,GenJet_pt,Jet_genJetIdx)")
                                            .Define("JetGen_mass","TakeIdx(Jet_mass,GenJet_mass,Jet_genJetIdx)")
                                            .Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"nom")')
                                            .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"nom")')
                           )


        res[dataset]={"nominal":rdf_dict[dataset],
                        
                    "JESUp":(rdf_dict[dataset].Redefine("Jet_pt","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                              .Redefine("Jet_mass","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JESDown":(rdf_dict[dataset].Redefine("Jet_pt","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                                .Redefine("Jet_mass","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "JERUp":(rdf_dict[dataset].Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"up")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"up")')
                            ),
                    "JERDown":(rdf_dict[dataset].Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"down")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"down")')
                            ),
                    }
        
            
        #!Remember to add the new systematics to the syst_dict
        syst_dict={ "nominal":[],
                "JESUp":["Jet_pt","Jet_mass"],
                "JESDown":["Jet_pt","Jet_mass"],
                "JERUp":["Jet_pt","Jet_mass"],
                "JERDown":["Jet_pt","Jet_mass"],
                }



    return res,syst_dict
