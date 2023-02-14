#include <ROOT/RDataFrame.hxx>
#include <iostream>
using namespace ROOT;


#include <regex>

#include "../../../utils/CMSStyle/CMS_lumi.C"
#include "../../../utils/CMSStyle/CMS_lumi.h"
#include "../../../utils/CMSStyle/tdrstyle.C"

#include "../../../utils/DfUtils.h"
#include "../../../utils/HistUtils.h"

void muonSelection(std::string filename, std::string text, std::string imageSaveFolder) {
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";
    datasetText = text;

    ROOT::EnableImplicitMT();

    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");

    RDataFrame fileDF("Events", filename,
                      {"LHEPart_pdgId",
                       "nMuon",
                       "Muon_pt",
                       "Muon_eta",
                       "Muon_phi",
                       "Muon_charge",
                       "Muon_looseId",
                       "Muon_mediumId",
                       "Muon_tightId",
                       "Muon_pfIsoId",
                       "Jet_jetId",
                       "Jet_puId",
                       "Jet_pt",
                       "Jet_muonIdx1",
                       "Jet_btagDeepFlavB"});

    auto MuonsFromWDF = fileDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)");

    auto TriggeredMuonsDF = MuonsFromWDF.Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

    auto LooseMuonsDF = TriggeredMuonsDF.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
    auto MediumMuonsDF = TriggeredMuonsDF.Filter("Muon_mediumId[0] && Muon_pfIsoId[0]>2");
    auto TightMuonsDF = TriggeredMuonsDF.Filter("Muon_tightId[0] && Muon_pfIsoId[0]>3");

    int totalTriggeredMuons = TriggeredMuonsDF.Count().GetValue();

    int totalLooseMuons = LooseMuonsDF.Count().GetValue();
    int totalMediumMuons = MediumMuonsDF.Count().GetValue();
    int totalTightMuons = TightMuonsDF.Count().GetValue();

    // tight/medium/loose w.r.t both identification and isolation
    std::cout << "Total triggered muons: " << totalTriggeredMuons << std::endl;
    std::cout << "Total loose muons: " << totalLooseMuons << "  Fraction:" << (float)totalLooseMuons / totalTriggeredMuons << std::endl;
    std::cout << "Total medium muons: " << totalMediumMuons << "  Fraction:" << (float)totalMediumMuons / totalTriggeredMuons << std::endl;
    std::cout << "Total tight muons: " << totalTightMuons << "  Fraction:" << (float)totalTightMuons / totalTriggeredMuons << std::endl;

    // Using loose muons
    LooseMuonsDF = LooseMuonsDF.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_muonIdx1!=0");
    for (auto &name : LooseMuonsDF.GetColumnNames()) {
        if (regex_match(name, std::regex("Jet_.*"))) {
            LooseMuonsDF = LooseMuonsDF.Redefine(name, name + "[JetMask]");
        }
    }
    LooseMuonsDF = LooseMuonsDF.Redefine("nJet", "Jet_pt.size()");

    auto histLeadingJetsWithoutMuon_pt = LooseMuonsDF.Define("LeadingJetWithoutMuon_pt", "Jet_pt[0]").Histo1D({"LeadingJetsWithoutMuon_pt", "Leading", 100, 0, 300}, "LeadingJetWithoutMuon_pt");

    auto histSecondJetsWithoutMuon_pt = LooseMuonsDF.Define("SecondJetWithoutMuon_pt", "Jet_pt[1]").Histo1D({"SecondJetsWithoutMuon_pt", "Second", 100, 0, 300}, "SecondJetWithoutMuon_pt");

    auto histThirdJetsWithoutMuon_pt = LooseMuonsDF.Define("ThirdJetWithoutMuon_pt", "Jet_pt[2]").Histo1D({"ThirdJetsWithoutMuon_pt", "Third", 100, 0, 300}, "ThirdJetWithoutMuon_pt");

    auto histFourthJetsWithoutMuon_pt = LooseMuonsDF.Define("FourthJetWithoutMuon_pt", "Jet_pt[3]").Histo1D({"FourthJetsWithoutMuon_pt", "Fourth", 100, 0, 300}, "FourthJetWithoutMuon_pt");

    int cutPtJet20 = LooseMuonsDF.Filter("nJet>=4").Filter("Jet_pt[3]>20").Count().GetValue();
    int cutPtJet30 = LooseMuonsDF.Filter("nJet>=4").Filter("Jet_pt[3]>30").Count().GetValue();

    std::cout << "Total events with fourth jet pT>20: " << cutPtJet20 << "  Fraction of events: " << (float)cutPtJet20 / totalLooseMuons << std::endl;
    std::cout << "Total events with fourth jet pT>30: " << cutPtJet30 << "  Fraction of events: " << (float)cutPtJet30 / totalLooseMuons << std::endl;

    StackPlotter orderedPtJetsWithoutMuon({histLeadingJetsWithoutMuon_pt, histSecondJetsWithoutMuon_pt, histThirdJetsWithoutMuon_pt, histFourthJetsWithoutMuon_pt}, "OrdedJetsWithoutMuon p_{t}", "p_{t} [GeV]", imageSaveFolder + "/OrdedJetsWithoutMuon_pt.png");

    //! NON TORNA IL TAGLIO SUL PT.Devono essere 139631 eventi alla fine


    LooseMuonsDF = LooseMuonsDF.Filter("nJet>=4").Filter("Jet_pt[3]>20");
    LooseMuonsDF = LooseMuonsDF.Define("LeadingJetsWithoutMuon_bTagProb", "Reverse(Sort(Jet_btagDeepFlavB))");

    //! Ricorda che qui c'è il taglio sul btag leading
    auto histLeadingJetsWithoutMuon_bTagProb = LooseMuonsDF.Define("LeadingJetWithoutMuon_bTagProb", "LeadingJetsWithoutMuon_bTagProb[0]").Histo1D({"LeadingJetsWithoutMuon_bTagProb", "Leading", 100, 0, 1}, "LeadingJetWithoutMuon_bTagProb");

    auto histSecondJetsWithoutMuon_bTagProb = LooseMuonsDF.Define("SecondJetWithoutMuon_bTagProb", "LeadingJetsWithoutMuon_bTagProb[1]").Histo1D({"SecondJetsWithoutMuon_bTagProb", "Second", 100, 0, 1}, "SecondJetWithoutMuon_bTagProb");

    auto histThirdJetsWithoutMuon_bTagProb = LooseMuonsDF.Define("ThirdJetWithoutMuon_bTagProb", "LeadingJetsWithoutMuon_bTagProb[2]").Histo1D({"ThirdJetsWithoutMuon_bTagProb", "Third", 100, 0, 1}, "ThirdJetWithoutMuon_bTagProb");

    auto histFourthJetsWithoutMuon_bTagProb = LooseMuonsDF.Define("FourthJetWithoutMuon_bTagProb", "LeadingJetsWithoutMuon_bTagProb[3]").Histo1D({"FourthJetsWithoutMuon_bTagProb", "Fourth", 100, 0, 1}, "FourthJetWithoutMuon_bTagProb");

    StackPlotter orderedBtagJetsWithoutMuon({histLeadingJetsWithoutMuon_bTagProb, histSecondJetsWithoutMuon_bTagProb, histThirdJetsWithoutMuon_bTagProb, histFourthJetsWithoutMuon_bTagProb}, "JetsWithoutMuon Jet_btagDeepFlavB", "Probability", imageSaveFolder + "/OrdedJetsWithoutMuon_bTagProb.png", false, false, true);

    orderedBtagJetsWithoutMuon.SetMinYlog(100);

    std::vector<StackPlotter *> stackCollection{
        &orderedPtJetsWithoutMuon,
        &orderedBtagJetsWithoutMuon};

    for (auto v : stackCollection) {
        v->Save();
    }
    exit(0);
}
