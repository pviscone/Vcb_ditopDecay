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

json_path="/scratchnvme/pviscone/Vcb_ditopDecay/tasks/combine/systematics/json"
#!JES
ROOT.gInterpreter.Declare(f'auto cset = correction::CorrectionSet::from_file("{json_path}/jet_jerc.json");')
ROOT.gInterpreter.Declare('auto JES = cset->at("Summer19UL18_V5_MC_Total_AK4PFchs");')


def vary(rdf_dict):
    syst_dict={"nominal":[],
                "upJES":["Jet_pt","Jet_mass"],
                "downJES":["Jet_pt","Jet_mass"]}

    res={}
    

    
    for dataset in rdf_dict:
        res[dataset]={"nominal":rdf_dict[dataset],
                        
                    "upJES":(rdf_dict[dataset].Redefine("Jet_pt","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                              .Redefine("Jet_mass","(1+evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            ),
                    
                    "downJES":(rdf_dict[dataset].Redefine("Jet_pt","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_pt")
                                                .Redefine("Jet_mass","(1-evaluate(JES,{Jet_eta,Jet_pt}))*Jet_mass")
                            )
                    }


    return res,syst_dict