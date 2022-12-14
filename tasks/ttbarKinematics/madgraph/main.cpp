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

std::unordered_map<std::string,float> CKM{
    {"ud",0.973},
    {"us",0.225},
    {"ub",0.0038},
    {"cd",0.221},
    {"cs",0.987},
    {"cb",0.041},
    {"td",0},
    {"ts",0},
    {"tb",0},
};

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
    gStyle->SetFillStyle(1001);

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

    RDataFrame fileDF("Events", "./73B85577-0234-814E-947E-7DCFC1275886.root",
                      {"nLHEPart",
                       "LHEPart_pt",
                       "LHEPart_eta",
                       "LHEPart_phi",
                       "LHEPart_mass",
                       "LHEPart_pdgId"});


    int nEvents = fileDF.Count().GetValue();

    // If you want to run on 10 event (for debugging purpose), uncomment, disable the MT and rename fileDF to fileDF10
    // auto fileDF=fileDF0.Range(10000);

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

    auto histMTLept = ptEtaPhiMDF.Histo1D({"histMTLept", "t#rightarrow bl#nu;M_{t} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TLept_mass");
    auto histMTHad = ptEtaPhiMDF.Histo1D({"histMTHad", "t#rightarrow bq#bar{q};M_{t}  [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "THad_mass");

    auto histMT = ptEtaPhiMDF.Histo1D({"histMT", "t;M_{t} [GeV]; Counts", nBinsTop, massTopmin, massTopmax}, "T_mass");
    auto histMTBar = ptEtaPhiMDF.Histo1D({"histMTBar", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TBar_mass");

    auto histMTLeptWide = ptEtaPhiMDF.Histo1D({"histMTLeptWide", "t#rightarrow l#nu;M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TLept_mass");
    auto histMTHadWide = ptEtaPhiMDF.Histo1D({"histMTHadWide", "t#rightarrow q#bar{q};M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "THad_mass");

    auto histMTWide = ptEtaPhiMDF.Histo1D({"histMTWide", "t;M_{t} [GeV]; Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "T_mass");
    auto histMTBarWide = ptEtaPhiMDF.Histo1D({"histMTBarWide", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TBar_mass");

    auto histMWLept = ptEtaPhiMDF.Histo1D({"histMWLept", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WLept_mass");
    auto histMWHad = ptEtaPhiMDF.Histo1D({"histMWHad", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WHad_mass");

    auto histMWPlus = ptEtaPhiMDF.Histo1D({"histMWPlus", "W^{+};M_{W^{+}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WPlus_mass");
    auto histMWMinus = ptEtaPhiMDF.Histo1D({"histMWMinus", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WMinus_mass");

    auto histMWLeptWide = ptEtaPhiMDF.Histo1D({"histMWLeptWide", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WLept_mass");
    auto histMWHadWide = ptEtaPhiMDF.Histo1D({"histMWHadWide", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WHad_mass");

    auto histMWPlusWide = ptEtaPhiMDF.Histo1D({"histMWPlusWide", "W^{+};M_{W^{+}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WPlus_mass");
    auto histMWMinusWide = ptEtaPhiMDF.Histo1D({"histMWMinusWide", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WMinus_mass");

    //-------------------------------------Pt-------------------------------------//
    double ptMin = 0;
    double ptMax = 500;
    int nBinsPt = 60;
    int nBinsPtSingle = 60;

    auto histPtTLept = ptEtaPhiMDF.Histo1D({"histPtTLept", "t#rightarrow bl#nu;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TLept_pt");
    auto histPtTHad = ptEtaPhiMDF.Histo1D({"histPtTHad", "t#rightarrow bq#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "THad_pt");

    auto histPtT = ptEtaPhiMDF.Histo1D({"histPtT", "t; p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "T_pt");
    auto histPtTBar = ptEtaPhiMDF.Histo1D({"histPtTBar", "#bar{t} ;p_{#bar{t}} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TBar_pt");

    auto histPtWLept = ptEtaPhiMDF.Histo1D({"histPtWLept", "W#rightarrow l#nu;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WLept_pt");
    auto histPtWHad = ptEtaPhiMDF.Histo1D({"histPtWHad", "W#rightarrow q#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WHad_pt");

    auto histPtWPlus = ptEtaPhiMDF.Histo1D({"histPtWPlus", "W^{+};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WPlus_pt");
    auto histPtWMinus = ptEtaPhiMDF.Histo1D({"histPtWMinus", "W^{-};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WMinus_pt");

    //! this section ptParticles is not future proof, it work only with this dataset (in which b=2,q=3,qbar=4,bbar=5,l=6)
    auto histPtB = ptEtaPhiMDF.Define("B_pt", "LHEPart_pt[2]").Histo1D({"histPtB", "b;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "B_pt");
    auto histPtBBar = ptEtaPhiMDF.Define("BBar_pt", "LHEPart_pt[5]").Histo1D({"histPtBBar", "#bar{b};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "BBar_pt");
    auto histPtQ = ptEtaPhiMDF.Define("Q_pt", "LHEPart_pt[3]").Histo1D({"histPtQ", "q;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Q_pt");
    auto histPtQBar = ptEtaPhiMDF.Define("QBar_pt", "LHEPart_pt[4]").Histo1D({"histPtQBar", "#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "QBar_pt");
    auto histPtLept = ptEtaPhiMDF.Define("Lept_pt", "LHEPart_pt[6]").Histo1D({"histPtL", "l;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Lept_pt");

    ptEtaPhiMDF=ptEtaPhiMDF.Define("Leading_pt", "leading(LHEPart_pt)");

    float ptMaxLeading = 350;

    auto histLeadingFirstPt = ptEtaPhiMDF.Define("Leading_firstPt","Leading_pt[0]").Histo1D({"histLeadingPt", "Leading p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_firstPt");

    auto histLeadingSecondPt = ptEtaPhiMDF.Define("Leading_secondPt", "Leading_pt[1]").Histo1D({"histLeadingSecondPt", "Second p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_secondPt");



    auto histLeadingThirdPt = ptEtaPhiMDF.Define("Leading_thirdPt", "Leading_pt[2]").Histo1D({"histLeadingThirdPt", "Third p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_thirdPt");

    auto histLeadingFourthPt = ptEtaPhiMDF.Define("Leading_fourthPt", "Leading_pt[3]").Histo1D({"histLeadingFourthPt", "Fourth p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_fourthPt");

    ptEtaPhiMDF=ptEtaPhiMDF.Define("Leading_ptPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_pt)");

    auto histLeadingFirstPtPdgId = ptEtaPhiMDF.Define("Leading_firstPtPdgId", "Leading_ptPdgId[0]").Histo1D({"histLeadingPtPdgId", "Leading;pdgId;Events", 2, 1, 3}, "Leading_firstPtPdgId");

    auto histLeadingSecondPtPdgId = ptEtaPhiMDF.Define("Leading_secondPtPdgId","Leading_ptPdgId[1]").Histo1D({"histLeadingSecondPtPdgId", "Second;pdgId;Events", 2, 1, 3}, "Leading_secondPtPdgId");

    auto histLeadingThirdPtPdgId = ptEtaPhiMDF.Define("Leading_thirdPtPdgId", "Leading_ptPdgId[2]").Histo1D({"histLeadingThirdPtPdgId", "Third;pdgId;Events", 2, 1, 3}, "Leading_thirdPtPdgId");

    auto histLeadingFourthPtPdgId = ptEtaPhiMDF.Define("Leading_fourthPtPdgId", "Leading_ptPdgId[3]").Histo1D({"histLeadingFourthPtPdgId", "Fourth;pdgId;Events", 2, 1, 3}, "Leading_fourthPtPdgId");

    // -------------------------------------Eta-------------------------------------//
    double etaMin = -6;
    double etaMax = 6;
    int nBinsEta = 50;
    int nBinsEtaSingle = 50;

    auto histEtaTLept = ptEtaPhiMDF.Histo1D({"histEtaTLept", "t#rightarrow bl#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "TLept_eta");
    auto histEtaTHad = ptEtaPhiMDF.Histo1D({"histEtaTHad", "t#rightarrow bq#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "THad_eta");

    auto histEtaT = ptEtaPhiMDF.Histo1D({"histEtaT", "t;#eta;Counts", nBinsEta, etaMin, etaMax}, "T_eta");
    auto histEtaTBar = ptEtaPhiMDF.Histo1D({"histEtaTBar", "#bar{t};#eta;Counts", nBinsEta, etaMin, etaMax}, "TBar_eta");

    auto histEtaWLept = ptEtaPhiMDF.Histo1D({"histEtaWLept", "W#rightarrow l#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "WLept_eta");
    auto histEtaWHad = ptEtaPhiMDF.Histo1D({"histEtaWHad", "W#rightarrow q#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "WHad_eta");

    auto histEtaWPlus = ptEtaPhiMDF.Histo1D({"histEtaWPlus", "W^{+};#eta;Counts", nBinsEta, etaMin, etaMax}, "WPlus_eta");
    auto histEtaWMinus = ptEtaPhiMDF.Histo1D({"histEtaWMinus", "W^{-};#eta;Counts", nBinsEta, etaMin, etaMax}, "WMinus_eta");

    ptEtaPhiMDF=ptEtaPhiMDF.Define("Leading_eta", "leading(LHEPart_eta,true)");
    
    auto histLeadingFirstEta =ptEtaPhiMDF.Define("Leading_firstEta","Leading_eta[0]").Histo1D({"histLeadingEta", "Leading #eta;#eta;Counts", nBinsEta, 0, etaMax}, "Leading_firstEta");

    auto histLeadingSecondEta = ptEtaPhiMDF.Define("Leading_secondEta", "Leading_eta[1]").Histo1D({"histLeadingSecondEta", "Second #eta;#eta;Counts", nBinsEta, 0, etaMax}, "Leading_secondEta");

    auto histLeadingThirdEta = ptEtaPhiMDF.Define("Leading_thirdEta", "Leading_eta[2]").Histo1D({"histLeadingThirdEta", "Third #eta;#eta;Counts", nBinsEta, 0, etaMax}, "Leading_thirdEta");

    auto histLeadingFourthEta = ptEtaPhiMDF.Define("Leading_fourthEta", "Leading_eta[3]").Histo1D({"histLeadingFourthEta", "Fourth #eta;#eta;Counts", nBinsEta, 0, etaMax}, "Leading_fourthEta");

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_etaPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_eta,true)");

    auto histLeadingFirstEtaPdgId = ptEtaPhiMDF.Define("Leading_firstEtaPdgId","Leading_etaPdgId[0]").Histo1D({"histLeadingEtaPdgId", "Leading;pdgId;Events", 2, 1, 3}, "Leading_firstEtaPdgId");

    auto histLeadingSecondEtaPdgId = ptEtaPhiMDF.Define("Leading_secondEtaPdgId", "Leading_etaPdgId[1]").Histo1D({"histLeadingSecondEtaPdgId", "Second;pdgId;Events", 2, 1, 3}, "Leading_secondEtaPdgId");

    auto histLeadingThirdEtaPdgId = ptEtaPhiMDF.Define("Leading_thirdEtaPdgId", "Leading_etaPdgId[2]").Histo1D({"histLeadingThirdEtaPdgId", "Third;pdgId;Events", 2, 1, 3}, "Leading_thirdEtaPdgId");

    auto histLeadingFourthEtaPdgId = ptEtaPhiMDF.Define("Leading_fourthEtaPdgId", "Leading_etaPdgId[3]").Histo1D({"histLeadingFourthEtaPdgId", "Fourth;pdgId;Events", 2, 1, 3}, "Leading_fourthEtaPdgId");

    //! this section etaParticles is not future proof, it work only with this dataset (in which b=2,q=3,qbar=4,bbar=5,l=6)
    auto histEtaB = ptEtaPhiMDF.Define("B_eta", "LHEPart_eta[2]").Histo1D({"histEtaB", "b;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "B_eta");
    auto histEtaQ = ptEtaPhiMDF.Define("Q_eta", "LHEPart_eta[3]").Histo1D({"histEtaQ", "q;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "Q_eta");
    auto histEtaQBar = ptEtaPhiMDF.Define("QBar_eta", "LHEPart_eta[4]").Histo1D({"histEtaQBar", "#bar{q};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "QBar_eta");
    auto histEtaBBar = ptEtaPhiMDF.Define("BBar_eta", "LHEPart_eta[5]").Histo1D({"histEtaBBar", "#bar{b};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "BBar_eta");
    auto histEtaLept = ptEtaPhiMDF.Define("Lept_eta", "LHEPart_eta[6]").Histo1D({"histEtaLept", "l;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "Lept_eta");

    //! NB This section (dEta,dPhi,dR) is not future proof, it is right only with this dataset in which all the WPlus decay hadronically and all the WMinus decay leptonically
    // The WPlus decays in u,c,dbar,sbar,bbar
    //-----------------------------------Delta eta-------------------------------------//
    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaEtaBQ", "LHEPart_eta[2]-LHEPart_eta[3]").Define("DeltaEtaBQBar", "LHEPart_eta[2]-LHEPart_eta[4]").Define("DeltaEtaBLept", "LHEPart_eta[2]-LHEPart_eta[6]").Define("DeltaEtaBBarQ", "LHEPart_eta[5]-LHEPart_eta[3]").Define("DeltaEtaBBarQBar", "LHEPart_eta[5]-LHEPart_eta[4]").Define("DeltaEtaBBarLept", "LHEPart_eta[5]-LHEPart_eta[6]");

    auto histDeltaEtaBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBU", "q;#Delta#eta;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQ");
    auto histDeltaEtaBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBDBar", "#bar{q};#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQBar");
    auto histDeltaEtaBLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBLept", "l;#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBLept");

    auto histDeltaEtaBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBBarQ", "q;#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQ");
    auto histDeltaEtaBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBBarQBar", "#bar{q};#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQBar");
    auto histDeltaEtaBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBBarLept", "l;#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarLept");

    //------------------------- Delta phi--------------------------------------
    int nBinsPhi = 60;
    int nBinsPhiB = 10;
    double phiMin = -3.14;
    double phiMax = 3.14;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaPhiBQ", "deltaPhi(LHEPart_phi[2],LHEPart_phi[3])").Define("DeltaPhiBQBar", "deltaPhi(LHEPart_phi[2],LHEPart_phi[4])").Define("DeltaPhiBLept", "deltaPhi(LHEPart_phi[2],LHEPart_phi[6])").Define("DeltaPhiBBarQ", "deltaPhi(LHEPart_phi[5],LHEPart_phi[3])").Define("DeltaPhiBBarQBar", "deltaPhi(LHEPart_phi[5],LHEPart_phi[4])").Define("DeltaPhiBBarLept", "deltaPhi(LHEPart_phi[5],LHEPart_phi[6])");

    auto histDeltaPhiBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBU", "q;#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBQ");

    auto histDeltaPhiBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBDBar", "#bar{q};#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBQBar");
    auto histDeltaPhiBLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBLept", "l;#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBLept");

    auto histDeltaPhiBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBBar", "q;#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarQ");
    auto histDeltaPhiBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBBarDBar", "#bar{q};#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarQBar");
    auto histDeltaPhiBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBBarLept", "l;#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarLept");

    //-------------------------------Delta R----------------------------------//
    int nBinsR = 60;
    int nBinsRB = 10;
    double RMin = 0;
    double RMax = 6.5;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaRBQ", "deltaR(DeltaPhiBQ,DeltaEtaBQ)").Define("DeltaRBQBar", "deltaR(DeltaPhiBQBar,DeltaEtaBQBar)").Define("DeltaRBLept", "deltaR(DeltaPhiBLept,DeltaEtaBLept)").Define("DeltaRBBarQ", "deltaR(DeltaPhiBBarQ,DeltaEtaBBarQ)").Define("DeltaRBBarQBar", "deltaR(DeltaPhiBBarQBar,DeltaEtaBBarQBar)").Define("DeltaRBBarLept", "deltaR(DeltaPhiBBarLept,DeltaEtaBBarLept)");

    auto histDeltaRBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBU", "q;#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBQ");
    auto histDeltaRBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBDBar", "#bar{q};#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBQBar");
    auto histDeltaRBLept = ptEtaPhiMDF.Histo1D({"histDeltaRBLept", "l;#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBLept");

    auto histDeltaRBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBBarU", "q;#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarQ");

    auto histDeltaRBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBBarDBar", "#bar{q};#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarQBar");
    auto histDeltaRBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaRBarBLept", "l;#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarLept");

    //----------------------------W hadronic decays----------------------------------//

    auto histWPlusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWPlus>0").Histo1D({"histWPlusJetDecay", "W^{+} qq decay; ;Counts", 9, 1, 9}, "jetCoupleWPlus");
/*     auto histWMinusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWMinus>0").Histo1D({"histWMinusJetDecay", "W^{-} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWMinus"); */

    StackPlotter jetCouple({histWPlusJetDecay}, "W hadronic Decays", "W qq Decay", "./images/WHadronicDecay.png", true, false, true);

    jetCouple.GetValue();
    TH1D* histCKM = new TH1D("CKM", "CKM Expected", 9, 1, 9);

    histCKM->SetBinContent(jetCoupleDictionary["ud"], nEvents*TMath::Power(CKM["ud"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["us"], nEvents*TMath::Power(CKM["us"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["ub"], nEvents*TMath::Power(CKM["ub"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["cd"], nEvents*TMath::Power(CKM["cd"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["cs"], nEvents*TMath::Power(CKM["cs"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["cb"], nEvents*TMath::Power(CKM["cb"],2)/2);
    histCKM->SetBinContent(jetCoupleDictionary["td"], 0);
    histCKM->SetBinContent(jetCoupleDictionary["ts"], 0);
    histCKM->SetBinContent(jetCoupleDictionary["tb"], 0);
    jetCouple.Add(histCKM);

    // Set the TH1 Label of the W decays the strings above
    (jetCouple).SetBinLabel(1, "ud");
    (jetCouple).SetBinLabel(2, "us");
    (jetCouple).SetBinLabel(3, "ub");
    (jetCouple).SetBinLabel(4, "cd");
    (jetCouple).SetBinLabel(5, "cs");
    (jetCouple).SetBinLabel(6, "cb");
    (jetCouple).SetBinLabel(7, "td");
    (jetCouple).SetBinLabel(8, "ts");
    (jetCouple).SetBinLabel(9, "tb");

    jetCouple.SetMaxStatBoxPrinted(1);



    //------------------------------------------DRAW---------------------------
    // (histvec,title,xlabel,path,ratio,fit,log)

    StackPlotter ttbarMass({histMTBar, histMT}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/Mttbar.png", true, true, false);
    StackPlotter tLeptHadMass({histMTHad, histMTLept}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHad.png", true, true);
    StackPlotter WPMMass({histMWPlus, histMWMinus}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinus.png", true, true);
    StackPlotter WLeptHadMass({histMWHad, histMWLept}, "M_{W#rightarrow q#bar{q} }/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHad.png", true, true);

    StackPlotter ttbarMassWide({histMTWide, histMTBarWide}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/MttbarWide.png", true);
    StackPlotter tLeptHadMassWide({histMTHadWide, histMTLeptWide}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHadWide.png", true);
    StackPlotter WPMMassWide({histMWPlusWide, histMWMinusWide}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinusWide.png", true);
    StackPlotter WLeptHadMassWide({histMWHadWide, histMWLeptWide}, "M_{W#rightarrow q#bar{q}}/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHadWide.png", true);

    StackPlotter ttbarEta({histEtaT, histEtaTBar}, "#eta_{t}/#eta_{#bar{t}}", "#eta_{t}", "./images/eta/EtaTTbar.png", true);
    StackPlotter tLeptHadEta({histEtaTHad, histEtaTLept}, "#eta_{t#rightarrow q#bar{q}} / #eta_{t#rightarrow l#nu}", "#eta_{t}", "./images/eta/EtaTLeptHad.png", true);
    StackPlotter WPMEta({histEtaWPlus, histEtaWMinus}, "#eta_{W^{+}}/#eta_{W^{-}}", "#eta_{W}", "./images/eta/EtaWPlusMinux.png", true);
    StackPlotter WLeptHadEta({histEtaWHad, histEtaWLept}, "#eta_{W#rightarrow q#bar{q}}/#eta_{W#rightarrow l#nu}", "#eta_{W}", "./images/eta/EtaWLeptHad.png", true);

    StackPlotter ttbarPt({histPtT, histPtTBar}, "p_{T}(t)/p_{T}(#bar{t})", "p_{T} [GeV]", "./images/pt/PtTTBar.png", true);
    StackPlotter tLeptHadPt({histPtTHad, histPtTLept}, "p_{T}(t#rightarrow q#bar{q})/p_{T}(t#rightarrow l#nu)", "p_{T} [GeV]", "./images/pt/PtTLeptHad.png", true);
    StackPlotter WPMPt({histPtWPlus, histPtWMinus}, "p_{T}(W^{+})/p_{T}(W^{-})", "p_{T} [GeV]", "./images/pt/PtWPlusMinus.png", true);
    StackPlotter WLeptHadPt({histPtWHad, histPtWLept}, "p_{T}(W#rightarrow q#bar{q})/p_{T}(W#rightarrow l#nu)", "p_{T} [GeV]", "./images/pt/PtWLeptHad.png", true);

    StackPlotter etaParticles({histEtaB, histEtaBBar, histEtaQ, histEtaQBar, histEtaLept}, "#eta", "#eta", "./images/eta/etaParticles.png");

    etaParticles.SetLegendPos({0.78, 0.6, 0.9, 0.9});
    etaParticles.SetPalette(55);
    etaParticles.SetDrawOpt("hist PMC PLC nostack");
    etaParticles.SetLineWidth(3);

    StackPlotter ptParticles({histPtB, histPtBBar, histPtQ, histPtQBar, histPtLept}, "p_{T}", "p_{T} [GeV]", "./images/pt/ptParticles.png");

    ptParticles.SetLegendPos({0.7, 0.6, 0.9, 0.9});
    ptParticles.SetPalette(55);
    ptParticles.SetDrawOpt("hist PMC PLC nostack");
    ptParticles.SetLineWidth(3);

    StackPlotter deltaEtaB({histDeltaEtaBQUp, histDeltaEtaBQDown, histDeltaEtaBLept}, "#Delta#eta b", "#Delta#eta", "./images/eta/deltaEtaB.png");
    StackPlotter deltaEtaBBar({histDeltaEtaBBarQUp, histDeltaEtaBBarQDown, histDeltaEtaBBarLept}, "#Delta#eta #bar{b}", "#Delta#eta", "./images/eta/deltaEtaBBar.png");

    StackPlotter deltaPhiB({histDeltaPhiBQUp, histDeltaPhiBQDown, histDeltaPhiBLept}, "#Delta#phi b", "#Delta#phi", "./images/phi/deltaPhiB.png");

    deltaPhiB.SetLegendPos({0.2, 0.74, 0.33, 0.86});
    StackPlotter deltaPhiBBar({histDeltaPhiBBarQUp, histDeltaPhiBBarQDown, histDeltaPhiBBarLept}, "#Delta#phi #bar{b}", "#Delta#phi ", "./images/phi/deltaPhiBBar.png");

    deltaPhiBBar.SetLegendPos({0.2, 0.74, 0.33, 0.86});
    StackPlotter deltaRB({histDeltaRBQUp, histDeltaRBQDown, histDeltaRBLept}, "#DeltaR b", "#DeltaR", "./images/r/deltaRB.png");

    StackPlotter deltaRBBar({histDeltaRBBarQUp, histDeltaRBBarQDown, histDeltaRBBarLept}, "#DeltaR #bar{b}", "#DeltaR", "./images/r/deltaRBBar.png");

    StackPlotter leadingPt({histLeadingFirstPt,histLeadingSecondPt,histLeadingThirdPt,histLeadingFourthPt}, "Leading p_{T}", "p_{T} [GeV]", "./images/pt/leadingPt.png");
    StackPlotter leadingPtPdgId({histLeadingFirstPtPdgId, histLeadingSecondPtPdgId, histLeadingThirdPtPdgId, histLeadingFourthPtPdgId}, "Leading p_{T} pdgId", "pdgId", "./images/pt/leadingPtpdgId.png");

    leadingPtPdgId.setQuarkTypeLabel();
    leadingPtPdgId.SetDrawOpt("bar");
    leadingPtPdgId.SetStatsInLegend(false);

    StackPlotter leadingEta({histLeadingFirstEta, histLeadingSecondEta, histLeadingThirdEta, histLeadingFourthEta}, "Leading #eta", "#eta", "./images/eta/leadingEta.png");
    StackPlotter leadingEtaPdgId({histLeadingFirstEtaPdgId, histLeadingSecondEtaPdgId, histLeadingThirdEtaPdgId, histLeadingFourthEtaPdgId}, "Leading #eta pdgId", "pdgId", "./images/eta/leadingEtapdgId.png");
    leadingEtaPdgId.setQuarkTypeLabel();
    leadingEtaPdgId.SetDrawOpt("bar");
    leadingEtaPdgId.SetStatsInLegend(false);

    std::vector<StackPlotter *> stackCollection{
     &ttbarMass,
        &ttbarMassWide,
        &ttbarEta,
        &ttbarPt,
        &tLeptHadMass,
        &tLeptHadMassWide,
        &tLeptHadEta,
        &tLeptHadPt,
        &WPMMass,
        &WPMMassWide,
        &WPMPt,
        &WPMEta,
        &WLeptHadMass,
        &WLeptHadMassWide,
        &WLeptHadPt,
        &WLeptHadEta,
        &WLeptHadPt,

        &jetCouple,

        &etaParticles,
        &ptParticles,
        &deltaEtaB,
        &deltaEtaBBar,
        &deltaPhiB,
        &deltaPhiBBar,
        &deltaRB,
        &deltaRBBar,

        &leadingPt,
        &leadingPtPdgId,

        &leadingEta,
        &leadingEtaPdgId

};

    for (auto v : stackCollection) {
        v->Save();
    }

    //-------------------------------------------------------------------------------
    //                                      DONE
    return 0;
}
