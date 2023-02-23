#include <ROOT/RDataFrame.hxx>
#include <iostream>
#include <regex>
using namespace ROOT;


void saveMuonCuts(std::string filename){
    EnableImplicitMT();
    RDataFrame df("Events",filename);
    auto dfCuts = df.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)").Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

    int nMuons = dfCuts.Count().GetValue();
    std::cout << "Number of triggered events (after trigger cuts on leading muon): " << nMuons << std::endl;

    
                    
    dfCuts=dfCuts.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");

    int nLooseMuons = dfCuts.Count().GetValue();
    std::cout << "Number of events after loose cuts on leading muons: " << nLooseMuons << std::endl;

    dfCuts=dfCuts.Define("MuonMask","Muon_pt>26 && abs(Muon_eta)<2.4 && Muon_looseId && Muon_pfIsoId>1");

    for(auto &name: df.GetColumnNames()){
        if(regex_match(name, std::regex("Muon_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[MuonMask]");
        }
    }
    dfCuts = dfCuts.Redefine("nMuon", "Muon_pt.size()");
    dfCuts = dfCuts.Filter("nMuon>=1");

    int nMuonsAfterMask = dfCuts.Count().GetValue();
    std::cout << "Number of events after the cut on all the muons: " << nMuonsAfterMask << std::endl;


    dfCuts = dfCuts.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_muonIdx1!=0 && Jet_pt>20");


    for(auto &name: df.GetColumnNames()){
        if(regex_match(name, std::regex("Jet_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[JetMask]");
        }
    }
    dfCuts = dfCuts.Redefine("nJet", "Jet_pt.size()");
    dfCuts = dfCuts.Filter("nJet>=4");


    int nMuonsJetsFilter = dfCuts.Count().GetValue();
    std::cout << "Number of events after jet pt filtering: " << nMuonsJetsFilter << std::endl;

    // Medium cut on btagDeepFlavB
    dfCuts = dfCuts.Filter("Reverse(Sort(Jet_btagDeepFlavB))[0]>0.2783");

    int nMuonsBtagFilter = dfCuts.Count().GetValue();
    std::cout << "Number of events after btag filtering: " << nMuonsBtagFilter << std::endl;
    dfCuts.Snapshot("Events", filename.replace(filename.find(".root"), 5, "_MuonSelection.root"));
}
