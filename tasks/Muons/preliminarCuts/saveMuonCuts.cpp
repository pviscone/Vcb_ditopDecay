#include <ROOT/RDataFrame.hxx>
#include <iostream>
#include <regex>
using namespace ROOT;

#include "./muonUtils.h"


void saveMuonCuts(std::string filename){
    RDataFrame df("Events",filename);
    auto dfCuts = df.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)").Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

    int nMuons = dfCuts.Count().GetValue();
    std::cout << "Number of triggere muons: " << nMuons << std::endl;

    
                    
    dfCuts=dfCuts.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");

    int nLooseMuons = dfCuts.Count().GetValue();
    std::cout << "Number of loose muons: " << nLooseMuons << std::endl;


    dfCuts = dfCuts.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_muonIdx1!=0 && Jet_pt>20");


    for(auto &name: df.GetColumnNames()){
        if(regex_match(name, std::regex("Jet_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[JetMask]");
        }
    }
    dfCuts = dfCuts.Redefine("nJet", "Jet_pt.size()");
    dfCuts = dfCuts.Filter("nJet>=4");

    int nMuonsJetsFilter = dfCuts.Count().GetValue();
    std::cout << "Number muons (after jet pt filtering): " << nMuonsJetsFilter << std::endl;

    dfCuts = dfCuts.Filter("Reverse(Sort(Jet_btagDeepFlavB))[0]>0.1");

    int nMuonsBtagFilter = dfCuts.Count().GetValue();
    std::cout << "Number muons (after btag filtering): " << nMuonsBtagFilter << std::endl;
    //dfCuts.Snapshot("Events", filename.replace(filename.find(".root"), 5, "_MuonSelection.root"));
}
