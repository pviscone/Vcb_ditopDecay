#include <iostream>
#include <unordered_map>

#include <Math/Vector4D.h>
#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TDatabasePDG.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLeaf.h>
#include <TROOT.h>
#include <TTree.h>
#include <stdlib.h>

#include "muonUtils.h"

#include "../../utils/itertools/product.hpp"
#include "../../utils/itertools/zip.hpp"

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"

#include "../../utils/DfUtils.h"
#include "../../utils/HistUtils.h"

void muonPreselection(std::string filename, std::string text, std::string imageSaveFolder) {
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";
    datasetText = text;

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();

    ROOT::EnableImplicitMT();
    RDataFrame fileDF("Events", filename,
                      {"LHEPart_pdgId",
                      "nMuon",
                      "Muon_pt",
                      "Muon_eta",
                      "Muon_phi",
                      "Muon_charge",});

    auto muonsDF=fileDF.Filter("selectMuonEvents(LHEPart_pdgId)");

    auto histNmuon = muonsDF.Histo1D({"nMuon", "nMuon", 10, 0, 10}, "nMuon");

    StackPlotter nMuon({histNmuon},"nMuon","",imageSaveFolder+"/nMuon.png");

//-----------------------------------------------------

    auto histMuonPt = muonsDF.Histo1D({"Muon_pt", "Muon_pt", 100, 0, 100}, "Muon_pt");

    StackPlotter Pt({histMuonPt}, "Muon p_{t}", "p_{t} [GeV]", imageSaveFolder + "/muonPt.png");

    auto histLeadingMuonPt = muonsDF.Filter("nMuon>0").Define("LeadingPt", "Muon_pt[0]").Histo1D({"Leading Muon_pt", "Leading p_{T};p_{T} [GeV];Counts", 100, 0, 170}, "LeadingPt");

    auto histSecondMuonPt = muonsDF.Filter("nMuon>1").Define("SecondPt", "Muon_pt[1]").Histo1D({"Second Muon_pt", "Second p_{T};p_{T} [GeV];Counts", 100, 0, 170}, "SecondPt");

    auto histThirdMuonPt = muonsDF.Filter("nMuon>2").Define("ThirdPt", "Muon_pt[2]").Histo1D({"Third Muon_pt", "Third p_{T};p_{T} [GeV];Counts", 100, 0, 170}, "ThirdPt");

    auto histFourthMuonPt = muonsDF.Filter("nMuon>3").Define("FourthPt", "Muon_pt[3]").Histo1D({"Fourth Muon_pt", "Fourth p_{T};p_{T} [GeV];Counts", 100, 0, 170}, "FourthPt");

    StackPlotter OrderedPt({histLeadingMuonPt,histSecondMuonPt,histThirdMuonPt,histFourthMuonPt}, "Ordered muon p_{t}", "p_{t} [GeV]", imageSaveFolder + "/orderedMuonPt.png");

//-----------------------------------------------------
    muonsDF = muonsDF.Define("orderedMuonEta", "orderAbs(Muon_eta)");

    auto histMuonEta = muonsDF.Histo1D({"Muon_eta", "Muon_eta", 60,0,2.8}, "orderedMuonEta");

    StackPlotter Eta({histMuonEta}, "Muon #eta", "#eta", imageSaveFolder + "/muonEta.png");



    auto histLeadingMuonEta = muonsDF.Filter("nMuon>0").Define("LeadingEta", "orderedMuonEta[0]").Histo1D({"Leading Muon_eta", "Leading #eta;#eta;Counts", 60, 0,2.8}, "LeadingEta");

    auto histSecondMuonEta = muonsDF.Filter("nMuon>1").Define("SecondEta", "orderedMuonEta[1]").Histo1D({"Second Muon_eta", "Second #eta;#eta;Counts", 60, 0,2.8}, "SecondEta");

    auto histThirdMuonEta = muonsDF.Filter("nMuon>2").Define("ThirdEta", "orderedMuonEta[2]").Histo1D({"Third Muon_eta", "Third #eta;#eta;Counts", 60, 0,2.8}, "ThirdEta");

    auto histFourthMuonEta = muonsDF.Filter("nMuon>3").Define("FourthEta", "orderedMuonEta[3]").Histo1D({"Fourth Muon_eta", "Fourth #eta;#eta;Counts", 60, 0,2.8}, "FourthEta");

    StackPlotter OrderedEta({histLeadingMuonEta, histSecondMuonEta, histThirdMuonEta, histFourthMuonEta}, "Ordered muon #eta", "#eta", imageSaveFolder + "/orderedMuonEta.png");

    //------------------------------------------------------

    std::vector<StackPlotter *> stackCollection{
        &nMuon,
        &Pt,
        &OrderedPt,
        &Eta,
        &OrderedEta,};

    for (auto v : stackCollection) {
        v->Save();
    }
    exit(EXIT_SUCCESS);
}
