import ROOT
import correctionlib
correctionlib.register_pyroot_binding()

ROOT.EnableImplicitMT()


ROOT.gInterpreter.Declare("""
    template <typename T>
    ROOT::RVec<float> evaluate(T cset, std::vector<ROOT::RVec<float>> const& inputs) {
        int size = inputs[0].size();
        ROOT::RVec<float> out(size);
        for(int i = 0; i < size; i++) {
            std::vector<correction::Variable::Type> in;
            for(auto const& input : inputs) {
                in.push_back(input[i]);
            }
            out[i] = cset->evaluate(in);
        }
        return out;
    }
    """)

ROOT.gInterpreter.Declare("""
    template <typename T, typename S>
    ROOT::RVec<float> evaluate(T cset, ROOT::RVec<float> const& input,const S &name) {
        int size = input.size();
        ROOT::RVec<float> out(size);
        for(int i = 0; i < size; i++) {
            out[i] = cset->evaluate({input[i],name});
        }
        return out;
    }
    """)


ROOT.gInterpreter.Declare("""
    ROOT::RVec<float> TakeIdx(ROOT::RVec<float> const& input,ROOT::RVec<int> const& idxs) {
        int size = idxs.size();
        ROOT::RVec<float> out(size);
        for(int i = 0; i < size; i++) {
            out[i]=input[idxs[i]];
        }
        return out;
    }
    """)

json_path="/scratchnvme/pviscone/Vcb_ditopDecay/tasks/combine/systematics/json"
#!JES
ROOT.gInterpreter.Declare(f'auto cset = correction::CorrectionSet::from_file("json/jet_jerc.json");')
ROOT.gInterpreter.Declare('auto JES = cset->at("Summer19UL18_V5_MC_Total_AK4PFchs");')
ROOT.gInterpreter.Declare('auto JER= cset->at("Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs");')



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
