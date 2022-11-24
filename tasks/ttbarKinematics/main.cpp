#include <iostream>
#include <unordered_map>

#include <Math/Vector4D.h>
#include <TCanvas.h>
#include <TDatabasePDG.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLeaf.h>
#include <TROOT.h>
#include <TTree.h>

#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"
#include "utils.h"

using namespace ROOT::Math;


int main() {

    //-------------------------------------------------------------------------------------------------------
    //                                 set the tdr style (from CMS TWIKI)
    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");

    TH1::SetDefaultSumw2();

    //-------------------------------------------------------------------------------------------------------
    //                                 File,tree and branches status
    TFile *file = new TFile("ttbar.root");
    TTree *tree = (TTree *)file->Get("Events");

    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("LHEPart*", 1);

    int NofEvents = tree->GetEntries();

    //-------------------------------------------------------------------------------------------------------
    //                                         Create the histograms
    TH1F *histMTLept = new TH1F("histMTLept", "M_{t} leptonic;M_{t}\\  [GeV];Counts", 60, 140, 200);
    TH1F *histMTHad = new TH1F("histMTHad", "M_{t} hadronic;M_{t}\\  [GeV];Counts", 60, 140, 200);

    TH1F *histMTBarLept = new TH1F("histMTBarLept", "M_{\\bar{t}}\\ leptonic;M_{\\bar{t}} \\ [GeV];Counts", 60, 140, 200);
    TH1F *histMTBarHad = new TH1F("histMTBarHad", "M_{\\bar{t}}\\ hadronic; M_{\\bar{t}} \\ [GeV];Counts", 60, 140, 200);

    TH1F *histMWPlusLept = new TH1F("histMWPlusLept", "M_{W^{+}} leptonic;M_{W^{+}} [GeV];Counts", 100, 0, 140);
    TH1F *histMWPlusHad = new TH1F("histMWPlusHad", "M_{W^{+}} hadronic;M_{W^{+}} [GeV];Counts", 100, 0, 140);

    TH1F *histMWMinusLept = new TH1F("histMWMinusLept", "M_{W^{-}} leptonic;M_{W^{-}} [GeV];Counts", 100, 0, 140);
    TH1F *histMWMinusHad = new TH1F("histMWMinusHad", "M_{W^{-}} hadronic;M_{W^{-}} [GeV];Counts", 100, 0, 140);

    TH1F *histPtTLept = new TH1F("histPtTLept", "t p_{t} leptonic;p_{t} [GeV];Counts", 60, 0, 600);
    TH1F *histPtTHad = new TH1F("histPtTHad", "t p_{t} hadronic;p_{t} [GeV];Counts", 60, 0, 600);

    TH1F *histPtTBarLept = new TH1F("histPtTBarLept", "\\bar{t} p_{t} leptonic; p_{t} [GeV];Counts", 60, 0, 600);
    TH1F *histPtTBarHad = new TH1F("histPtTBarHad", "\\bar{t} p_{t} hadronic;p_{\\bar{t}} [GeV];Counts", 60, 0, 600);

    TH1F *histPtWPlusLept = new TH1F("histPtWPlusLept", "W^{+} p_{t} leptonic;p_{t} [GeV];Counts", 60, 0, 600);
    TH1F *histPtWPlusHad = new TH1F("histPtWPlusHad", "W^{+} p_{t} hadronic;p_{t} [GeV];Counts", 60, 0, 600);

    TH1F *histPtWMinusLept = new TH1F("histPtWMinusLept", "W^{-} p_{t} leptonic;p_{t} [GeV];Counts", 60, 0, 600);
    TH1F *histPtWMinusHad = new TH1F("histPtWMinusHad", "W^{-} p_{t} hadronic;p_{t} [GeV];Counts", 60, 0, 600);

    TH1F *histEtaTLept = new TH1F("histEtaTLept", "t \\eta leptonic;\\eta;Counts", 50, -10, 10);
    TH1F *histEtaTHad = new TH1F("histEtaTHad", "t \\eta hadronic;\\eta;Counts", 50, -10, 10);

    TH1F *histEtaTBarLept = new TH1F("histEtaTBarLept", "\\bar{t} \\eta leptonic;\\eta;Counts", 50, -10, 10);
    TH1F *histEtaTBarHad = new TH1F("histEtaTBarHad", "\\bar{t} \\eta hadronic;\\eta;Counts", 50, -10, 10);

    TH1F *histEtaWPlusLept = new TH1F("histEtaWPlusLept", "W^{+} \\eta leptonic;\\eta;Counts", 50, -10, 10);
    TH1F *histEtaWPlusHad = new TH1F("histEtaWPlusHad", "W^{+} \\eta hadronic;\\eta;Counts", 50, -10, 10);

    TH1F *histEtaWMinusLept = new TH1F("histEtaWMinusLept", "W^{-} \\eta leptonic;\\eta;Counts", 50, -10, 10);
    TH1F *histEtaWMinusHad = new TH1F("histEtaWMinusHad", "W^{-} \\eta hadronic;\\eta;Counts", 50, -10, 10);

    TH1I *histWPlusJetDecay = new TH1I("histWPlusJetDecay", "W^{+} jet decay; ;Counts", 10, 1, 10);
    TH1I *histWMinusJetDecay = new TH1I("histWMinusJetDecay", "W^{-} jet decay; ;Counts", 10, 1, 10);

    //-------------------------------------------------------------------------------------------------------
    //         Create a dictionary to assign to each qq couple a unique number (for the histograms)
    std::unordered_map<std::string, int> jetCoupleDictionary;
    jetCoupleDictionary["tb"] = 9;
    jetCoupleDictionary["ts"] = 8;
    jetCoupleDictionary["td"] = 7;
    jetCoupleDictionary["cb"] = 6;
    jetCoupleDictionary["cs"] = 5;
    jetCoupleDictionary["cd"] = 4;
    jetCoupleDictionary["ub"] = 3;
    jetCoupleDictionary["us"] = 2;
    jetCoupleDictionary["ud"] = 1;

    // Set the TH1 Label of the W decays the strings above
    for (auto &couple : jetCoupleDictionary) {
        histWPlusJetDecay->GetXaxis()->SetBinLabel(couple.second, couple.first.c_str());
        histWMinusJetDecay->GetXaxis()->SetBinLabel(couple.second, couple.first.c_str());
    }

    //Add to the dictionary also with the reversed characters to make a symmetric dictionary
    jetCoupleDictionary["du"] = 1;
    jetCoupleDictionary["su"] = 2;
    jetCoupleDictionary["bu"] = 3;
    jetCoupleDictionary["dc"] = 4;
    jetCoupleDictionary["sc"] = 5;
    jetCoupleDictionary["bc"] = 6;
    jetCoupleDictionary["dt"] = 7;
    jetCoupleDictionary["st"] = 8;
    jetCoupleDictionary["bt"] = 9;

    //-------------------------------------------------------------------------------------------------------
    //      Define a Lorentz vectors for each particles

    //This vector will contain index (Instance) of the particles produced by the W+ and by the W-
    std::vector<int> indexFromWPlus(2);
    std::vector<int> indexFromWMinus(2);
    //The particles in the 3rd instance comes always from the top, the 4rth from the antitop
    int indexQFromT = 3;
    int indexQBarFromTBar = 4;

    PtEtaPhiMVector lorentzVectorWPlusLept;
    PtEtaPhiMVector lorentzVectorWPlusHad;
    PtEtaPhiMVector lorentzVectorWMinusLept;
    PtEtaPhiMVector lorentzVectorWMinusHad;
    PtEtaPhiMVector lorentzVectorTLept;
    PtEtaPhiMVector lorentzVectorTHad;
    PtEtaPhiMVector lorentzVectorTBarLept;
    PtEtaPhiMVector lorentzVectorTBarHad;


    // NofEvents=100000;
    //-------------------------------------------------------------------------------------------------------
    //                                       Loop over the events
    for (int eventNumber = 0; eventNumber < NofEvents; eventNumber++) {
        tree->GetEntry(eventNumber);

        // The charge of the 5th particle is in agreement with the charge of the W that produced it
        // The 5th and 6th particles are always produced together (same for the 7th and 8th)
        int pdgId5 = (tree->GetLeaf("LHEPart_pdgId"))->GetValue(5);
        if (particle(pdgId5)->Charge() > 0.) {
            indexFromWPlus[0] = 5;
            indexFromWPlus[1] = 6;
            indexFromWMinus[0] = 7;
            indexFromWMinus[1] = 8;
        } else {
            indexFromWPlus[0] = 7;
            indexFromWPlus[1] = 8;
            indexFromWMinus[0] = 5;
            indexFromWMinus[1] = 6;
        }
        // Discriminate the hadronic and the leptonic decay and fill the histograms
        // (The possible processes are t->q-W+ and tbar->qbar+W-)
        std::string jetCouple;
        if (isQuark(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[0]))) {
            lorentzVectorWPlusHad = getLorentzVector(tree, indexFromWPlus[0]) + getLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWMinusLept = getLorentzVector(tree, indexFromWMinus[0]) + getLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTBarLept = getLorentzVector(tree, indexQBarFromTBar) + lorentzVectorWMinusLept;
            lorentzVectorTHad = getLorentzVector(tree, indexQFromT) + lorentzVectorWPlusHad;

            histMTBarLept->Fill(lorentzVectorTBarLept.M());
            histEtaTBarLept->Fill(lorentzVectorTBarLept.Eta());
            histPtTBarLept->Fill(lorentzVectorTBarLept.Pt());

            histMTHad->Fill(lorentzVectorTHad.M());
            histEtaTHad->Fill(lorentzVectorTHad.Eta());
            histPtTHad->Fill(lorentzVectorTHad.Pt());

            histMWPlusHad->Fill(lorentzVectorWPlusHad.M());
            histEtaWPlusHad->Fill(lorentzVectorWPlusHad.Eta());
            histPtWPlusHad->Fill(lorentzVectorWPlusHad.Pt());

            histMWMinusLept->Fill(lorentzVectorWMinusLept.M());
            histEtaWMinusLept->Fill(lorentzVectorWMinusLept.Eta());
            histPtWMinusLept->Fill(lorentzVectorWMinusLept.Pt());

            std::string quark1 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[0]));
            std::string quark2 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[1]));
            jetCouple.push_back(quark1[0]);
            jetCouple.push_back(quark2[0]);

            histWPlusJetDecay->Fill(jetCoupleDictionary[jetCouple]);

        } else {
            lorentzVectorWPlusLept = getLorentzVector(tree, indexFromWPlus[0]) + getLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWMinusHad = getLorentzVector(tree, indexFromWMinus[0]) + getLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTBarHad = getLorentzVector(tree, indexQBarFromTBar) + lorentzVectorWMinusHad;
            lorentzVectorTLept = getLorentzVector(tree, indexQFromT) + lorentzVectorWPlusLept;

            histMTBarHad->Fill(lorentzVectorTBarHad.M());
            histEtaTBarHad->Fill(lorentzVectorTBarHad.Eta());
            histPtTBarHad->Fill(lorentzVectorTBarHad.Pt());

            histMTLept->Fill(lorentzVectorTLept.M());
            histEtaTLept->Fill(lorentzVectorTLept.Eta());
            histPtTLept->Fill(lorentzVectorTLept.Pt());

            histMWPlusLept->Fill(lorentzVectorWPlusLept.M());
            histEtaWPlusLept->Fill(lorentzVectorWPlusLept.Eta());
            histPtWPlusLept->Fill(lorentzVectorWPlusLept.Pt());

            histMWMinusHad->Fill(lorentzVectorWMinusHad.M());
            histEtaWMinusHad->Fill(lorentzVectorWMinusHad.Eta());
            histPtWMinusHad->Fill(lorentzVectorWMinusHad.Pt());

            std::string quark1 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWMinus[0]));
            std::string quark2 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWMinus[1]));
            jetCouple.push_back(quark1[0]);
            jetCouple.push_back(quark2[0]);
            histWMinusJetDecay->Fill(jetCoupleDictionary[jetCouple]);
        }
    }

    //-------------------------------------------------------------------------------------------------------
    //                                      Draw the histograms
    StackHist(histMTHad, histMTLept, "Top invariant mass", "M_{t}  [GeV]", "./images/mass/Mt.png");
    StackHist(histMTBarHad, histMTBarLept, "TBar Invariant mass", "M_{\bar{t}} [GeV]", "./images/mass/Mtbar.png");
    StackHist(histMWPlusHad, histMWPlusLept, "W^{+} Invariant mass", "M_{W^{+}} [GeV]", "./images/mass/MWPlus.png");
    StackHist(histMWMinusHad, histMWMinusLept, "W^{-} Invariant mass", "M_{W^{-}} [GeV]", "./images/mass/MWMinus.png");

    StackHist(histEtaTHad, histEtaTLept, "Top Eta", "\\eta_{t}", "./images/eta/EtaT.png");
    StackHist(histEtaTBarHad, histEtaTBarLept, "TBar Eta", "\\eta_{\bar{t}}", "./images/eta/EtaTbar.png");
    StackHist(histEtaWPlusHad, histEtaWPlusLept, "W^{+} Eta", "\\e\\ta_{W^{+}}", "./images/eta/EtaWPlus.png");
    StackHist(histEtaWMinusHad, histEtaWMinusLept, "W^{-} eta", "\\eta_{W^{-}}", "./images/eta/EtaWMinus.png");

    StackHist(histPtTHad, histPtTLept, "Top P_{t}", "p_{t} [GeV]", "./images/pt/Pt.png");
    StackHist(histPtTBarHad, histPtTBarLept, "TBar P_{t}", "p_{t} [GeV]", "./images/pt/Ptbar.png");
    StackHist(histPtWPlusHad, histPtWPlusLept, "W^{+} P_{t}", "p_{t} [GeV]", "./images/pt/PtWPlus.png");
    StackHist(histPtWMinusHad, histPtWMinusLept, "W^{-} P_{t}", "p_{t} [GeV]", "./images/pt/PtWMinus.png");

    StackHist(histWPlusJetDecay, histWMinusJetDecay, "W^{+} Jet Decay", "W^{+} Decay", "./images/WJetDecay.png");

    //-------------------------------------------------------------------------------------------------------
    //                                      DONE
    return 0;
}
