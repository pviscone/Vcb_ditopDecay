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

#include "../../../utils/CMSStyle/CMS_lumi.C"
#include "../../../utils/CMSStyle/CMS_lumi.h"
#include "../../../utils/CMSStyle/tdrstyle.C"
#include "DfUtils.h"
#include "HistUtils.h"

using namespace ROOT;
using namespace ROOT::Math;
using namespace ROOT::RDF;

// This vector will contain index (Instance) of the particles produced by the W+ and by the W-
std::vector<int> indexFromWPlus{3, 4};
std::vector<int> indexFromWMinus{6, 7};
// The particles in the 3rd instance comes always from the top, the 4rth from the antitop
int indexQFromT = 2;
int indexQBarFromTBar = 5;

std::unordered_map<std::string, int> jetCoupleDictionary{
    {"tb", 9},
    {"ts", 8},
    {"td", 7},
    {"cb", 6},
    {"cs", 5},
    {"cd", 4},
    {"ub", 3},
    {"us", 2},
    {"ud", 1},
    {"du", 1},
    {"su", 2},
    {"bu", 3},
    {"dc", 4},
    {"sc", 5},
    {"bc", 6},
    {"dt", 7},
    {"st", 8},
    {"bt", 9}};

int jetCoupleWPlus(RVec<int> pdgIdVec) {

    if (isQuark(pdgIdVec[indexFromWPlus[0]])) {
        std::string key;
        std::string q1 = pdg(pdgIdVec[indexFromWPlus[0]]);
        std::string q2 = pdg(pdgIdVec[indexFromWPlus[1]]);
        key.push_back(q1[0]);
        key.push_back(q2[0]);
        return jetCoupleDictionary[key];
    } else {
        return -1;
    }
}

int jetCoupleWMinus(RVec<int> pdgIdVec) {

    if (isQuark(pdgIdVec[indexFromWMinus[0]])) {
        std::string key;
        std::string q1 = pdg(pdgIdVec[indexFromWMinus[0]]);
        std::string q2 = pdg(pdgIdVec[indexFromWMinus[1]]);
        key.push_back(q1[0]);
        key.push_back(q2[0]);
        std::cout << isQuark(pdgIdVec[indexFromWMinus[0]]) << std::endl;
        return jetCoupleDictionary[key];
    } else {
        return -1;
    }
}

float isHadronic(float isQuark, float sameSign, float oppositeSign) {
    if (isQuark >= 0) {
        return sameSign;
    } else {
        return oppositeSign;
    }
}

float isLeptonic(float isQuark, float sameSign, float oppositeSign) {
    if (isQuark < 0) {
        return sameSign;
    } else {
        return oppositeSign;
    }
}

int main() {

    //-------------------------------------------------------------------------------------------------------
    //                                 set the tdr style (from CMS TWIKI)
    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();

    // Draw "Preliminary"
    writeExtraText = false;

    //-------------------------------------------------------------------------------------------------------
    //                                 File,tree and branches status

    RDataFrame fileDF("Events", "./0BCB1429-8B19-3245-92C4-68B3DD50AC78.root",
                      {"nLHEPart",
                       "LHEPart_pt",
                       "LHEPart_eta",
                       "LHEPart_phi",
                       "LHEPart_mass",
                       "LHEPart_pdgId"});

    // If you want to run on 10 event (for debugging purpose), uncomment, disable the MT and rename fileDF to fileDF10
    // auto fileDF=fileDF0.Range(10);

    //-------------------------------------------------------------------------------------------------------
    //      Define a Lorentz vectors for each particles

    auto lorentzVectorsDF = fileDF
                                .Define("WMinus", PtEtaPhiMVecSum(indexFromWMinus))
                                .Define("WPlus", PtEtaPhiMVecSum(indexFromWPlus));

    lorentzVectorsDF = lorentzVectorsDF
                           .Define("T", PtEtaPhiMVecSum({indexQFromT}, "WPlus"))
                           .Define("TBar", PtEtaPhiMVecSum({indexQBarFromTBar}, "WMinus"));

    auto ptEtaPhiMDF = lorentzVectorsDF
                           .Define("WMinus_pt", "WMinus.pt()")
                           .Define("WMinus_eta", "WMinus.eta()")
                           .Define("WMinus_phi", "WMinus.phi()")
                           .Define("WMinus_mass", "WMinus.mass()")
                           .Define("WPlus_pt", "WPlus.pt()")
                           .Define("WPlus_eta", "WPlus.eta()")
                           .Define("WPlus_phi", "WPlus.phi()")
                           .Define("WPlus_mass", "WPlus.mass()")
                           .Define("T_pt", "T.pt()")
                           .Define("T_eta", "T.eta()")
                           .Define("T_phi", "T.phi()")
                           .Define("T_mass", "T.mass()")
                           .Define("TBar_pt", "TBar.pt()")
                           .Define("TBar_eta", "TBar.eta()")
                           .Define("TBar_phi", "TBar.phi()")
                           .Define("TBar_mass", "TBar.mass()")
                           .Define("jetCoupleWPlus", "jetCoupleWPlus(LHEPart_pdgId)")
                           .Define("jetCoupleWMinus", "jetCoupleWMinus(LHEPart_pdgId)");

    ptEtaPhiMDF = ptEtaPhiMDF
                      .Define("WLept_mass", "isLeptonic(jetCoupleWPlus,WPlus_mass,WMinus_mass)")
                      .Define("WLept_pt", "isLeptonic(jetCoupleWPlus,WPlus_pt,WMinus_pt)")
                      .Define("WLept_eta", "isLeptonic(jetCoupleWPlus,WPlus_eta,WMinus_eta)")
                      .Define("WLept_phi", "isLeptonic(jetCoupleWPlus,WPlus_phi,WMinus_phi)")
                      .Define("WHad_mass", "isHadronic(jetCoupleWPlus,WPlus_mass,WMinus_mass)")
                      .Define("WHad_pt", "isHadronic(jetCoupleWPlus,WPlus_pt,WMinus_pt)")
                      .Define("WHad_eta", "isHadronic(jetCoupleWPlus,WPlus_eta,WMinus_eta)")
                      .Define("WHad_phi", "isHadronic(jetCoupleWPlus,WPlus_phi,WMinus_phi)")
                      .Define("TLept_mass", "isLeptonic(jetCoupleWPlus,T_mass,TBar_mass)")
                      .Define("TLept_pt", "isLeptonic(jetCoupleWPlus,T_pt,TBar_pt)")
                      .Define("TLept_eta", "isLeptonic(jetCoupleWPlus,T_eta,TBar_eta)")
                      .Define("TLept_phi", "isLeptonic(jetCoupleWPlus,T_phi,TBar_phi)")
                      .Define("THad_mass", "isHadronic(jetCoupleWPlus,T_mass,TBar_mass)")
                      .Define("THad_pt", "isHadronic(jetCoupleWPlus,T_pt,TBar_pt)")
                      .Define("THad_eta", "isHadronic(jetCoupleWPlus,T_eta,TBar_eta)")
                      .Define("THad_phi", "isHadronic(jetCoupleWPlus,T_phi,TBar_phi)");

    //-------------------------------------------------------------------------------------------------------
    //                                          Create the histograms

    //-----------------------------Masses-----------------------------//
    double massW = 80.385;
    double massTop = 172.5;
    double widthW = 2.085;
    double widthTop = 1.41;
    double plotWidthMultiplier = 6.;

    double wideMultiplier = 4.;
    double binWideMultiplier = 2.;

    double massWmin = massW - plotWidthMultiplier * widthW;
    double massWmax = massW + plotWidthMultiplier * widthW;
    double massTopmin = massTop - plotWidthMultiplier * widthTop;
    double massTopmax = massTop + plotWidthMultiplier * widthTop;

    double massWminWide = massW - plotWidthMultiplier * widthW * wideMultiplier;
    double massWmaxWide = massW + plotWidthMultiplier * widthW * wideMultiplier;
    double massTopminWide = massTop - plotWidthMultiplier * widthTop * wideMultiplier;
    double massTopmaxWide = massTop + plotWidthMultiplier * widthTop * wideMultiplier;

    int nBinsTop = 4 * (2 * plotWidthMultiplier * widthTop);
    int nBinsW = 4 * (2 * plotWidthMultiplier * widthW);
    int nBinsTopWide = nBinsTop * binWideMultiplier;
    int nBinsWWide = nBinsW * binWideMultiplier;

    // non si sa perchè ma getptr è rottissimpo. usa getvalue e poi passa gli indirizzi

    TH1D histMTLept = ptEtaPhiMDF.Histo1D({"histMTLept", "t#rightarrow bl#nu;M_{t} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TLept_mass").GetValue();
    TH1D histMTHad = ptEtaPhiMDF.Histo1D({"histMTHad", "t#rightarrow bq#bar{q};M_{t}  [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "THad_mass").GetValue();

    TH1D histMT = ptEtaPhiMDF.Histo1D({"histMT", "t;M_{t} [GeV]; Counts", nBinsTop, massTopmin, massTopmax}, "T_mass").GetValue();
    TH1D histMTBar = ptEtaPhiMDF.Histo1D({"histMTBar", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TBar_mass").GetValue();

    TH1D histMTLeptWide = ptEtaPhiMDF.Histo1D({"histMTLeptWide", "t#rightarrow l#nu;M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TLept_mass").GetValue();
    TH1D histMTHadWide = ptEtaPhiMDF.Histo1D({"histMTHadWide", "t#rightarrow q#bar{q};M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "THad_mass").GetValue();

    TH1D histMTWide = ptEtaPhiMDF.Histo1D({"histMTWide", "t;M_{t} [GeV]; Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "T_mass").GetValue();
    TH1D histMTBarWide = ptEtaPhiMDF.Histo1D({"histMTBarWide", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TBar_mass").GetValue();

    TH1D histMWLept = ptEtaPhiMDF.Histo1D({"histMWLept", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WLept_mass").GetValue();
    TH1D histMWHad = ptEtaPhiMDF.Histo1D({"histMWHad", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WHad_mass").GetValue();

    TH1D histMWPlus = ptEtaPhiMDF.Histo1D({"histMWPlus", "W^{+};M_{W^{+}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WPlus_mass").GetValue();
    TH1D histMWMinus = ptEtaPhiMDF.Histo1D({"histMWMinus", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WMinus_mass").GetValue();

    TH1D histMWLeptWide = ptEtaPhiMDF.Histo1D({"histMWLeptWide", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WLept_mass").GetValue();
    TH1D histMWHadWide = ptEtaPhiMDF.Histo1D({"histMWHadWide", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WHad_mass").GetValue();

    TH1D histMWPlusWide = ptEtaPhiMDF.Histo1D({"histMWPlusWide", "W^{+};M_{W^{+}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WPlus_mass").GetValue();
    TH1D histMWMinusWide = ptEtaPhiMDF.Histo1D({"histMWMinusWide", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WMinus_mass").GetValue();

    //-------------------------------------Pt-------------------------------------//
    double ptMin = 0;
    double ptMax = 500;
    int nBinsPt = 60;

    TH1D histPtTLept = ptEtaPhiMDF.Histo1D({"histPtTLept", "t#rightarrow bl#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TLept_pt").GetValue();
    TH1D histPtTHad = ptEtaPhiMDF.Histo1D({"histPtTHad", "t#rightarrow bq#bar{q};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "THad_pt").GetValue();

    TH1D histPtT = ptEtaPhiMDF.Histo1D({"histPtT", "t; p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "T_pt").GetValue();
    TH1D histPtTBar = ptEtaPhiMDF.Histo1D({"histPtTBar", "#bar{t} ;p_{#bar{t}} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TBar_pt").GetValue();

    TH1D histPtWLept = ptEtaPhiMDF.Histo1D({"histPtWLept", "W#rightarrow l#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WLept_pt").GetValue();
    TH1D histPtWHad = ptEtaPhiMDF.Histo1D({"histPtWHad", "W#rightarrow q#bar{q};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WHad_pt").GetValue();

    TH1D histPtWPlus = ptEtaPhiMDF.Histo1D({"histPtWPlus", "W^{+};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WPlus_pt").GetValue();
    TH1D histPtWMinus = ptEtaPhiMDF.Histo1D({"histPtWMinus", "W^{-};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WMinus_pt").GetValue();

    // TODO define new hist and new quantities
    /*     TH1D histPtBFromT = ptEtaPhiMDF.Histo1D({"histPtBFromT", "b from t;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax},"").GetValue();
        TH1D histPtBBarFromTBar = ptEtaPhiMDF.Histo1D({"histPtBBarFromTBar", "b from #bar{t};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax},"").GetValue();
        TH1D histPtBBarFromWPlus = ptEtaPhiMDF.Histo1D({"histPtBBarFromWPlus", "b from W;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax},"").GetValue();
        TH1D histPtC = ptEtaPhiMDF.Histo1D({"histPtC", "c;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax},"").GetValue();
        TH1D histPtLept = ptEtaPhiMDF.Histo1D({"histPtLept", "lepton;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax},"").GetValue(); */

    //-------------------------------------Eta-------------------------------------//
    double etaMin = -6;
    double etaMax = 6;
    int nBinsEta = 50;

    TH1D histEtaTLept = ptEtaPhiMDF.Histo1D({"histEtaTLept", "t#rightarrow bl#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "TLept_eta").GetValue();
    TH1D histEtaTHad = ptEtaPhiMDF.Histo1D({"histEtaTHad", "t#rightarrow bq#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "THad_eta").GetValue();

    TH1D histEtaT = ptEtaPhiMDF.Histo1D({"histEtaT", "t;#eta;Counts", nBinsEta, etaMin, etaMax}, "T_eta").GetValue();
    TH1D histEtaTBar = ptEtaPhiMDF.Histo1D({"histEtaTBar", "#bar{t};#eta;Counts", nBinsEta, etaMin, etaMax}, "TBar_eta").GetValue();

    TH1D histEtaWLept = ptEtaPhiMDF.Histo1D({"histEtaWLept", "W#rightarrow l#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "WLept_eta").GetValue();
    TH1D histEtaWHad = ptEtaPhiMDF.Histo1D({"histEtaWHad", "W#rightarrow q#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "WHad_eta").GetValue();

    TH1D histEtaWPlus = ptEtaPhiMDF.Histo1D({"histEtaWPlus", "W^{+};#eta;Counts", nBinsEta, etaMin, etaMax}, "WPlus_eta").GetValue();
    TH1D histEtaWMinus = ptEtaPhiMDF.Histo1D({"histEtaWMinus", "W^{-};#eta;Counts", nBinsEta, etaMin, etaMax}, "WMinus_eta").GetValue();

    // TODO define new hist and new quantities
    /*     TH1D histEtaBFromT = ptEtaPhiMDF.Histo1D({"histEtaB", "b;#eta;Counts", nBinsEta, etaMin, etaMax},"").GetValue();
        TH1D histEtaBBarFromTBar = ptEtaPhiMDF.Histo1D({"histEtaBBar", "#bar{b};#eta;Counts", nBinsEta, etaMin, etaMax},"").GetValue();
        TH1D histEtaBBarFromWPlus = ptEtaPhiMDF.Histo1D({"histEtaBBarFromWPlus", "b;#eta;Counts", nBinsEta, etaMin, etaMax},"").GetValue();
        TH1D histEtaC = ptEtaPhiMDF.Histo1D({"histEtaC", "c;#eta;Counts", nBinsEta, etaMin, etaMax},"").GetValue();
        TH1D histEtaLep = ptEtaPhiMDF.Histo1D({"histEtaLep", "l;#eta;Counts", nBinsEta, etaMin, etaMax},"").GetValue(); */

    //-----------------------------------Delta eta-------------------------------------//

    //----------------------------W hadronic decays----------------------------------//

    TH1D histWPlusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWPlus>0").Histo1D({"histWPlusJetDecay", "W^{+} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWPlus").GetValue();
    TH1D histWMinusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWMinus>0").Histo1D({"histWMinusJetDecay", "W^{-} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWMinus").GetValue();

    // Set the TH1 Label of the W decays the strings above
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(1, "ud");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(1, "ud");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(2, "us");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(2, "us");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(3, "ub");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(3, "ub");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(4, "cd");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(4, "cd");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(5, "cs");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(5, "cs");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(6, "cb");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(6, "cb");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(7, "td");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(7, "td");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(8, "ts");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(8, "ts");
    (&histWPlusJetDecay)->GetXaxis()->SetBinLabel(9, "tb");
    (&histWMinusJetDecay)->GetXaxis()->SetBinLabel(9, "tb");

    //------------------------------------------DRAW---------------------------
    // (histvec,title,xlabel,path,ratio,fit,log)

    StackHist({&histMTBar, &histMT}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/Mttbar.png", true, true, false);

    StackHist({&histMT, &histMTBar}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/Mttbar.png", true, true);
    StackHist({&histMTHad, &histMTLept}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHad.png", true, true);
    StackHist({&histMWPlus, &histMWMinus}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinus.png", true, true);
    StackHist({&histMWHad, &histMWLept}, "M_{W#rightarrow q#bar{q} }/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHad.png", true, true);

    StackHist({&histMTWide, &histMTBarWide}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/MttbarWide.png", true);
    StackHist({&histMTHadWide, &histMTLeptWide}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHadWide.png", true);
    StackHist({&histMWPlusWide, &histMWMinusWide}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinusWide.png", true);
    StackHist({&histMWHadWide, &histMWLeptWide}, "M_{W#rightarrow q#bar{q}}/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHadWide.png", true);

    StackHist({&histEtaT, &histEtaTBar}, "#eta_{t}/#eta_{#bar{t}}", "#eta_{t}", "./images/eta/EtaTTbar.png", true);
    StackHist({&histEtaTHad, &histEtaTLept}, "#eta_{t#rightarrow q#bar{q}} / #eta_{t#rightarrow l#nu}", "#eta_{t}", "./images/eta/EtaTLeptHad.png", true);
    StackHist({&histEtaWPlus, &histEtaWMinus}, "#eta_{W^{+}}/#eta_{W^{-}}", "#eta_{W}", "./images/eta/EtaWPlusMinux.png", true);
    StackHist({&histEtaWHad, &histEtaWLept}, "#eta_{W#rightarrow q#bar{q}}/#eta_{W#rightarrow l#nu}", "#eta_{W}", "./images/eta/EtaWLeptHad.png", true);

    StackHist({&histPtT, &histPtTBar}, "p_{t}(t)/p_{t}(#bar{t})", "p_{t} [GeV]", "./images/pt/PtTTBar.png", true);
    StackHist({&histPtTHad, &histPtTLept}, "p_{t}(t#rightarrow q#bar{q})/p_{t}(t#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtTLeptHad.png", true);
    StackHist({&histPtWPlus, &histPtWMinus}, "p_{t}(W^{+})/p_{t}(W^{-})", "p_{t} [GeV]", "./images/pt/PtWPlusMinus.png", true);
    StackHist({&histPtWHad, &histPtWLept}, "p_{t}(W#rightarrow q#bar{q})/p_{t}(W#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtWLeptHad.png", true);

    // in this dataset all the Wplus are hadronic and all the Wminus leptonic
    StackHist({&histWPlusJetDecay}, "W hadronic Decays", "W qq Decay", "./images/WHadronicDecay.png", false, false, true);

    //-------------------------------------------------------------------------------------------------------
    //                                      DONE
    return 0;
}
