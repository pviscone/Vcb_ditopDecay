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
#include "../../utils/CMSStyle/CMS_lumi.C"
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

    //Draw "Preliminary"
    writeExtraText = false;

    //-------------------------------------------------------------------------------------------------------
    //                                 File,tree and branches status
    TFile *file = new TFile("ttbar.root");
    TTree *tree = (TTree *)file->Get("Events");

    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("LHEPart*", 1);

    int NofEvents = tree->GetEntries();

    //-------------------------------------------------------------------------------------------------------
    //                                         Create the histograms


    //-----------------------------Masses-----------------------------//
    double massW=80.385;
    double massTop=172.5;
    double widthW=2.085;
    double widthTop=1.41;
    double plotWidthMultiplier=6.;

    double wideMultiplier=4.;
    double binWideMultiplier=2.;

    double massWmin=massW-plotWidthMultiplier*widthW;
    double massWmax=massW+plotWidthMultiplier*widthW;
    double massTopmin=massTop-plotWidthMultiplier*widthTop;
    double massTopmax=massTop+plotWidthMultiplier*widthTop;

    double massWminWide = massW - plotWidthMultiplier * widthW*wideMultiplier;
    double massWmaxWide = massW + plotWidthMultiplier * widthW*wideMultiplier;
    double massTopminWide = massTop - plotWidthMultiplier * widthTop*wideMultiplier;
    double massTopmaxWide = massTop + plotWidthMultiplier * widthTop*wideMultiplier;


    int nBinsTop=4*(2*plotWidthMultiplier*widthTop);
    int nBinsW=4*(2*plotWidthMultiplier*widthW);
    int nBinsTopWide = nBinsTop*binWideMultiplier;
    int nBinsWWide = nBinsW*binWideMultiplier;

    TH1F *histMTLept = new TH1F("histMTLept", "t#rightarrow l#nu;M_{t} [GeV];Counts", nBinsTop, massTopmin, massTopmax);
    TH1F *histMTHad = new TH1F("histMTHad", "t#rightarrow q#bar{q};M_{t}  [GeV];Counts", nBinsTop, massTopmin, massTopmax);

    TH1F *histMT = new TH1F("histMT", "t;M_{t} [GeV]; Counts", nBinsTop, massTopmin, massTopmax);
    TH1F *histMTBar = new TH1F("histMTBar", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTop, massTopmin, massTopmax);

    TH1F *histMTLeptWide = new TH1F("histMTLeptWide", "t#rightarrow l#nu;M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide);
    TH1F *histMTHadWide = new TH1F("histMTHadWide", "t#rightarrow q#bar{q};M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide);

    TH1F *histMTWide = new TH1F("histMTWide", "t;M_{t} [GeV]; Counts", nBinsTopWide, massTopminWide, massTopmaxWide);
    TH1F *histMTBarWide = new TH1F("histMTBarWide", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide);

    TH1F *histMWLept = new TH1F("histMWLept", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsW, massWmin, massWmax);
    TH1F *histMWHad = new TH1F("histMWHad", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsW, massWmin, massWmax);

    TH1F *histMWPlus = new TH1F("histMWPlus", "W^{+};M_{W^{+}} [GeV];Counts", nBinsW, massWmin, massWmax);
    TH1F *histMWMinus = new TH1F("histMWMinus", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsW, massWmin, massWmax);

    TH1F *histMWLeptWide = new TH1F("histMWLeptWide", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide);
    TH1F *histMWHadWide = new TH1F("histMWHadWide", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide);

    TH1F *histMWPlusWide = new TH1F("histMWPlusWide", "W^{+};M_{W^{+}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide);
    TH1F *histMWMinusWide = new TH1F("histMWMinusWide", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide);

    //-------------------------------------Pt-------------------------------------//
    double ptMin=0;
    double ptMax=500;
    int nBinsPt=60;

    TH1F *histPtTLept = new TH1F("histPtTLept", "t#rightarrow l#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);
    TH1F *histPtTHad = new TH1F("histPtTHad", "t#rightarrow q#bar{q};p_{t} [GeV];Counts",nBinsPt, ptMin, ptMax);

    TH1F *histPtT = new TH1F("histPtT", "t; p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);
    TH1F *histPtTBar = new TH1F("histPtTBar", "#bar{t} ;p_{#bar{t}} [GeV];Counts", nBinsPt, ptMin, ptMax);

    TH1F *histPtWLept = new TH1F("histPtWLept", "W#rightarrow l#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);
    TH1F *histPtWHad = new TH1F("histPtWHad", "W#rightarrow q#bar{q};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);

    TH1F *histPtWPlus = new TH1F("histPtWPlus", "W^{+};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);
    TH1F *histPtWMinus = new TH1F("histPtWMinus", "W^{-};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax);

    //-------------------------------------Eta-------------------------------------//
    double etaMin=-6;
    double etaMax=6;
    int nBinsEta=50;

    TH1F *histEtaTLept = new TH1F("histEtaTLept", "t#rightarrow l#nu;#eta;Counts", nBinsEta, etaMin, etaMax);
    TH1F *histEtaTHad = new TH1F("histEtaTHad", "t#rightarrow q#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax);

    TH1F *histEtaT = new TH1F("histEtaT", "t;#eta;Counts", nBinsEta, etaMin, etaMax);
    TH1F *histEtaTBar= new TH1F("histEtaTBar", "#bar{t};#eta;Counts", nBinsEta, etaMin, etaMax);

    TH1F *histEtaWLept = new TH1F("histEtaWLept", "W#rightarrow l#nu;#eta;Counts", nBinsEta, etaMin, etaMax);
    TH1F *histEtaWHad = new TH1F("histEtaWHad", "W#rightarrow q#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax);

    TH1F *histEtaWPlus = new TH1F("histEtaWPlus", "W^{+};#eta;Counts", nBinsEta, etaMin, etaMax);
    TH1F *histEtaWMinus = new TH1F("histEtaWMinus", "W^{-};#eta;Counts", nBinsEta, etaMin, etaMax);

    //----------------------------W hadronic decays----------------------------------//

    TH1F *histWPlusJetDecay = new TH1F("histWPlusJetDecay", "W^{+} jet decay; ;Counts", 9, 1, 9);
    TH1F *histWMinusJetDecay = new TH1F("histWMinusJetDecay", "W^{-} jet decay; ;Counts", 9, 1, 9);

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

    //Add to the dictionary also the reversed characters to make a symmetric dictionary
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

    PtEtaPhiMVector lorentzVectorWPlus;
    PtEtaPhiMVector lorentzVectorWMinus;
    PtEtaPhiMVector lorentzVectorWLept;
    PtEtaPhiMVector lorentzVectorWHad;
    PtEtaPhiMVector lorentzVectorTLept;
    PtEtaPhiMVector lorentzVectorTHad;
    PtEtaPhiMVector lorentzVectorT;
    PtEtaPhiMVector lorentzVectorTBar;


    //NofEvents=1000;
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

        lorentzVectorWPlus = getLorentzVector(tree, indexFromWPlus[0]) + getLorentzVector(tree, indexFromWPlus[1]);
        lorentzVectorWMinus = getLorentzVector(tree, indexFromWMinus[0]) + getLorentzVector(tree, indexFromWMinus[1]);
        lorentzVectorTBar = getLorentzVector(tree, indexQBarFromTBar) + lorentzVectorWMinus;
        lorentzVectorT = getLorentzVector(tree, indexQFromT) + lorentzVectorWPlus;

        // Discriminate the hadronic and the leptonic decay and fill the histograms
        // (The possible processes are t->q-W+ and tbar->qbar+W-)
        std::string jetCouple;
        if (isQuark(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[0]))) {
            lorentzVectorWHad = getLorentzVector(tree, indexFromWPlus[0]) + getLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWLept = getLorentzVector(tree, indexFromWMinus[0]) + getLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTLept = getLorentzVector(tree, indexQBarFromTBar) + lorentzVectorWLept;
            lorentzVectorTHad = getLorentzVector(tree, indexQFromT) + lorentzVectorWHad;

            std::string quark1 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[0]));
            std::string quark2 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[1]));
            jetCouple.push_back(quark1[0]);
            jetCouple.push_back(quark2[0]);

            histWPlusJetDecay->Fill(jetCoupleDictionary[jetCouple]);

        } else {
            lorentzVectorWLept = getLorentzVector(tree, indexFromWPlus[0]) + getLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWHad = getLorentzVector(tree, indexFromWMinus[0]) + getLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTHad = getLorentzVector(tree, indexQBarFromTBar) + lorentzVectorWHad;
            lorentzVectorTLept = getLorentzVector(tree, indexQFromT) + lorentzVectorWLept;

            std::string quark1 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWMinus[0]));
            std::string quark2 = pdg(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWMinus[1]));
            jetCouple.push_back(quark1[0]);
            jetCouple.push_back(quark2[0]);
            histWMinusJetDecay->Fill(jetCoupleDictionary[jetCouple]);
        }

        //-------------------------------------------Fill the histograms----------------------------------------
        //Masses t (also wide)
        histMT->Fill(lorentzVectorT.M());
        histMTBar->Fill(lorentzVectorTBar.M());
        histMTHad->Fill(lorentzVectorTHad.M());
        histMTLept->Fill(lorentzVectorTLept.M());
        histMTWide->Fill(lorentzVectorT.M());
        histMTBarWide->Fill(lorentzVectorTBar.M());
        histMTHadWide->Fill(lorentzVectorTHad.M());
        histMTLeptWide->Fill(lorentzVectorTLept.M());
        //Masses W
        histMWPlus->Fill(lorentzVectorWPlus.M());
        histMWMinus->Fill(lorentzVectorWMinus.M());
        histMWHad->Fill(lorentzVectorWHad.M());
        histMWLept->Fill(lorentzVectorWLept.M());
        histMWPlusWide->Fill(lorentzVectorWPlus.M());
        histMWMinusWide->Fill(lorentzVectorWMinus.M());
        histMWHadWide->Fill(lorentzVectorWHad.M());
        histMWLeptWide->Fill(lorentzVectorWLept.M());

        //Pt t
        histPtT->Fill(lorentzVectorT.Pt());
        histPtTBar->Fill(lorentzVectorTBar.Pt());
        histPtTHad->Fill(lorentzVectorTHad.Pt());
        histPtTLept->Fill(lorentzVectorTLept.Pt());
        //Pt W
        histPtWPlus->Fill(lorentzVectorWPlus.Pt());
        histPtWMinus->Fill(lorentzVectorWMinus.Pt());
        histPtWHad->Fill(lorentzVectorWHad.Pt());
        histPtWLept->Fill(lorentzVectorWLept.Pt());

        //Eta t
        histEtaT->Fill(lorentzVectorT.Eta());
        histEtaTBar->Fill(lorentzVectorTBar.Eta());
        histEtaTHad->Fill(lorentzVectorTHad.Eta());
        histEtaTLept->Fill(lorentzVectorTLept.Eta());
        //Eta W
        histEtaWPlus->Fill(lorentzVectorWPlus.Eta());
        histEtaWMinus->Fill(lorentzVectorWMinus.Eta());
        histEtaWHad->Fill(lorentzVectorWHad.Eta());
        histEtaWLept->Fill(lorentzVectorWLept.Eta());
    }

    //-------------------------------------------------------------------------------------------------------
    //                                      Draw the histograms
    StackHist(histMT, histMTBar, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/Mttbar.png",true);
    StackHist(histMTHad, histMTLept, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHad.png",true);
    StackHist(histMWPlus, histMWMinus, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinus.png",true);
    StackHist(histMWHad, histMWLept, "M_{W#rightarrow q#bar{q} }/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHad.png",true);

    StackHist(histMTWide, histMTBarWide, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/MttbarWide.png");
    StackHist(histMTHadWide, histMTLeptWide, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHadWide.png");
    StackHist(histMWPlusWide, histMWMinusWide, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinusWide.png");
    StackHist(histMWHadWide, histMWLeptWide, "M_{W#rightarrow q#bar{q}}/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHadWide.png");

    StackHist(histEtaT, histEtaTBar, "#eta_{t}/#eta_{#bar{t}}", "#eta_{t}", "./images/eta/EtaTTbar.png");
    StackHist(histEtaTHad, histEtaTLept, "#eta_{t#rightarrow q#bar{q}} / #eta_{t#rightarrow l#nu}", "#eta_{t}", "./images/eta/EtaTLeptHad.png");
    StackHist(histEtaWPlus, histEtaWMinus, "#eta_{W^{+}}/#eta_{W^{-}}", "#eta_{W}", "./images/eta/EtaWPlusMinux.png");
    StackHist(histEtaWHad, histEtaWLept, "#eta_{W#rightarrow q#bar{q}}/#eta_{W#rightarrow l#nu}", "#eta_{W}", "./images/eta/EtaWLeptHad.png");


    StackHist(histPtT, histPtTBar, "p_{t}(t)/p_{t}(#bar{t})", "p_{t} [GeV]", "./images/pt/PtTTBar.png");
    StackHist(histPtTHad, histPtTLept, "p_{t}(t#rightarrow q#bar{q})/p_{t}(t#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtTLeptHad.png");
    StackHist(histPtWPlus, histPtWMinus, "p_{t}(W^{+})/p_{t}(W^{-})", "p_{t} [GeV]", "./images/pt/PtWPlusMinus.png");
    StackHist(histPtWHad, histPtWLept, "p_{t}(W#rightarrow q#bar{q})/p_{t}(W#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtWLeptHad.png");


    StackHist(histWPlusJetDecay, histWMinusJetDecay, "W hadronic Decays", "W qq Decay", "./images/WHadronicDecay.png");

    //-------------------------------------------------------------------------------------------------------
    //                                      DONE
    return 0;
}
