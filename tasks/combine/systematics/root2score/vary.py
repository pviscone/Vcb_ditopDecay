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
    
    #!Remember to add the new systematics to the syst_dict
    syst_dict={"nominal":[],
                "upJES":["Jet_pt","Jet_mass"],
                "downJES":["Jet_pt","Jet_mass"],
                "upJER":["Jet_pt","Jet_mass"],
                "downJER":["Jet_pt","Jet_mass"],
                "nomJER":["Jet_pt","Jet_mass"],
                }

    res={}
    

    
    for dataset in rdf_dict:
        rdf_dict[dataset]=(rdf_dict[dataset].Define("JetGen_pt","TakeIdx(GenJet_pt,Jet_genJetIdx)")
                                            .Define("JetGen_mass","TakeIdx(GenJet_mass,Jet_genJetIdx)")
                           )


        res[dataset]={"nominal":rdf_dict[dataset],
                        
                    "upJES":(rdf_dict[dataset].Redefine("Jet_pt","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                              .Redefine("Jet_mass","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "downJES":(rdf_dict[dataset].Redefine("Jet_pt","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                                .Redefine("Jet_mass","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    "nomJER":(rdf_dict[dataset].Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"nom")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"nom")')
                            ),
                    "upJER":(rdf_dict[dataset].Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"up")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"up")')
                            ),
                    "downJER":(rdf_dict[dataset].Redefine("Jet_pt",'JetGen_pt+(Jet_pt-JetGen_pt)*evaluate(JER,{Jet_eta},"down")')
                                                .Redefine("Jet_mass",'JetGen_mass+(Jet_mass-JetGen_mass)*evaluate(JER,{Jet_eta},"down")')
                            ),
                    }


    return res,syst_dict
