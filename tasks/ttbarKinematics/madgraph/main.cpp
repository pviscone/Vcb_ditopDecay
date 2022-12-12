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

    RDataFrame fileDF("Events", "./0BCB1429-8B19-3245-92C4-68B3DD50AC78.root",
                      {"nLHEPart",
                       "LHEPart_pt",
                       "LHEPart_eta",
                       "LHEPart_phi",
                       "LHEPart_mass",
                       "LHEPart_pdgId"});

    // If you want to run on 10 event (for debugging purpose), uncomment, disable the MT and rename fileDF to fileDF10
    //auto fileDF=fileDF0.Range(10000);

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
    int nBinsPtSingle=30;

    auto histPtTLept = ptEtaPhiMDF.Histo1D({"histPtTLept", "t#rightarrow bl#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TLept_pt");
    auto histPtTHad = ptEtaPhiMDF.Histo1D({"histPtTHad", "t#rightarrow bq#bar{q};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "THad_pt");

    auto histPtT = ptEtaPhiMDF.Histo1D({"histPtT", "t; p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "T_pt");
    auto histPtTBar = ptEtaPhiMDF.Histo1D({"histPtTBar", "#bar{t} ;p_{#bar{t}} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TBar_pt");

    auto histPtWLept = ptEtaPhiMDF.Histo1D({"histPtWLept", "W#rightarrow l#nu;p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WLept_pt");
    auto histPtWHad = ptEtaPhiMDF.Histo1D({"histPtWHad", "W#rightarrow q#bar{q};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WHad_pt");

    auto histPtWPlus = ptEtaPhiMDF.Histo1D({"histPtWPlus", "W^{+};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WPlus_pt");
    auto histPtWMinus = ptEtaPhiMDF.Histo1D({"histPtWMinus", "W^{-};p_{t} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WMinus_pt");



    auto histPtB = fileDF.Define("B_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==5) && LHEPart_status==1)").Histo1D({"histPtB", "b;p_{t}[GeV];   Counts",  nBinsPtSingle, ptMin, ptMax}, "B_pt");
    auto histPtBBar = fileDF.Define("BBar_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==-5) && LHEPart_status==1)").Histo1D({"histPtBBar", "#bar{b};p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "BBar_pt");
    auto histPtC = fileDF.Define("C_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==4) && LHEPart_status==1)").Histo1D({"histPtC", "c;p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "C_pt");
    auto histPtCBar = fileDF.Define("CBar_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==-4) && LHEPart_status==1)").Histo1D({"histPtCBar", "#bar{c};p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "CBar_pt");
    auto histPtS = fileDF.Define("S_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==3) && LHEPart_status==1)").Histo1D({"histPtS", "s;p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "S_pt");
    auto histPtSBar = fileDF.Define("SBar_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==-3) && LHEPart_status==1)").Histo1D({"histPtSBar", "#bar{s};p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "SBar_pt");
    auto histPtD = fileDF.Define("D_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==1) && LHEPart_status==1)").Histo1D({"histPtD", "d;p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "D_pt");
    auto histPtDBar = fileDF.Define("DBar_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==-1) && LHEPart_status==1)").Histo1D({"histPtDBar", "#bar{d};p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "DBar_pt");
    auto histPtU = fileDF.Define("U_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==2) && LHEPart_status==1)").Histo1D({"histPtU", "u;p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "U_pt");
    auto histPtUBar = fileDF.Define("UBar_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==-2) && LHEPart_status==1)").Histo1D({"histPtUBar", "#bar{u};p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "UBar_pt");
    auto histPtLept = fileDF.Define("Lept_pt", "filterTracks(LHEPart_pt,(LHEPart_pdgId==11 || LHEPart_pdgId==13 || LHEPart_pdgId==15) && LHEPart_status==1)").Histo1D({"histPtLept", "leptons;p_{t}[GeV];Counts",  nBinsPtSingle, ptMin, ptMax}, "Lept_pt");

    // -------------------------------------Eta-------------------------------------//
    double etaMin = -6;
    double etaMax = 6;
    int nBinsEta = 50;
    int nBinsEtaB = 20;
    int nBinsEtaSingle = 25;

    auto histEtaTLept = ptEtaPhiMDF.Histo1D({"histEtaTLept", "t#rightarrow bl#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "TLept_eta");
    auto histEtaTHad = ptEtaPhiMDF.Histo1D({"histEtaTHad", "t#rightarrow bq#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "THad_eta");

    auto histEtaT = ptEtaPhiMDF.Histo1D({"histEtaT", "t;#eta;Counts", nBinsEta, etaMin, etaMax}, "T_eta");
    auto histEtaTBar = ptEtaPhiMDF.Histo1D({"histEtaTBar", "#bar{t};#eta;Counts", nBinsEta, etaMin, etaMax}, "TBar_eta");

    auto histEtaWLept = ptEtaPhiMDF.Histo1D({"histEtaWLept", "W#rightarrow l#nu;#eta;Counts", nBinsEta, etaMin, etaMax}, "WLept_eta");
    auto histEtaWHad = ptEtaPhiMDF.Histo1D({"histEtaWHad", "W#rightarrow q#bar{q};#eta;Counts", nBinsEta, etaMin, etaMax}, "WHad_eta");

    auto histEtaWPlus = ptEtaPhiMDF.Histo1D({"histEtaWPlus", "W^{+};#eta;Counts", nBinsEta, etaMin, etaMax}, "WPlus_eta");
    auto histEtaWMinus = ptEtaPhiMDF.Histo1D({"histEtaWMinus", "W^{-};#eta;Counts", nBinsEta, etaMin, etaMax}, "WMinus_eta");


    auto histEtaB=fileDF.Define("B_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==5) && LHEPart_status==1)").Histo1D({"histEtaB", "b;#eta;   Counts", nBinsEtaSingle, etaMin, etaMax}, "B_eta");
    auto histEtaBBar=fileDF.Define("BBar_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==-5) && LHEPart_status==1)").Histo1D({"histEtaBBar", "#bar{b};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "BBar_eta");
    auto histEtaC=fileDF.Define("C_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==4) && LHEPart_status==1)").Histo1D({"histEtaC", "c;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "C_eta");
    auto histEtaCBar=fileDF.Define("CBar_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==-4) && LHEPart_status==1)").Histo1D({"histEtaCBar", "#bar{c};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "CBar_eta");
    auto histEtaS=fileDF.Define("S_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==3) && LHEPart_status==1)").Histo1D({"histEtaS", "s;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "S_eta");
    auto histEtaSBar=fileDF.Define("SBar_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==-3) && LHEPart_status==1)").Histo1D({"histEtaSBar", "#bar{s};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "SBar_eta");
    auto histEtaD=fileDF.Define("D_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==1) && LHEPart_status==1)").Histo1D({"histEtaD", "d;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "D_eta");
    auto histEtaDBar=fileDF.Define("DBar_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==-1) && LHEPart_status==1)").Histo1D({"histEtaDBar", "#bar{d};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "DBar_eta");
    auto histEtaU=fileDF.Define("U_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==2) && LHEPart_status==1)").Histo1D({"histEtaU", "u;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "U_eta");
    auto histEtaUBar=fileDF.Define("UBar_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==-2) && LHEPart_status==1)").Histo1D({"histEtaUBar", "#bar{u};#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "UBar_eta");
    auto histEtaLept=fileDF.Define("Lept_eta","filterTracks(LHEPart_eta,(LHEPart_pdgId==11 || LHEPart_pdgId==13 || LHEPart_pdgId==15) && LHEPart_status==1)").Histo1D({"histEtaLept", "leptons;#eta;Counts", nBinsEtaSingle, etaMin, etaMax}, "Lept_eta");


    //! NB This section (dEta,dPhi,dR) is not future proof, it is right only with this dataset in which all the WPlus decay hadronically and all the WMinus decay leptonically
    //The WPlus decays in u,c,dbar,sbar,bbar
    //-----------------------------------Delta eta-------------------------------------//
    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaEtaBQ", "LHEPart_eta[2]-LHEPart_eta[3]").Define("DeltaEtaBQBar", "LHEPart_eta[2]-LHEPart_eta[4]").Define("DeltaEtaBLept", "LHEPart_eta[2]-LHEPart_eta[6]").Define("DeltaEtaBBarQ", "LHEPart_eta[5]-LHEPart_eta[3]").Define("DeltaEtaBBarQBar", "LHEPart_eta[5]-LHEPart_eta[4]").Define("DeltaEtaBBarLept", "LHEPart_eta[5]-LHEPart_eta[6]");

    auto histDeltaEtaBU = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaEtaBU", "u;#Delta#eta;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQ");
    auto histDeltaEtaBC = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBC", "c;#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQ");
    auto histDeltaEtaBDBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaEtaBDBar", "#bar{d};#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQ");
    auto histDeltaEtaBSBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaEtaBSBar", "#bar{s};#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBQBar");
    auto histDeltaEtaBBBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBBBar", "#bar{b};#Delta#eta b;Counts", nBinsEtaB, etaMin, etaMax}, "DeltaEtaBQBar");
    auto histDeltaEtaBLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBLept", "leptons;#Delta#eta b;Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBLept");

    auto histDeltaEtaBBarU = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaEtaBBarU", "u;#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQ");
    auto histDeltaEtaBBarC = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBBarC", "c;#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQ");
    auto histDeltaEtaBBarDBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaEtaBBarDBar", "#bar{d};#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQBar");
    auto histDeltaEtaBBarSBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaEtaBBarSBar", "#bar{s};#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarQBar");
    auto histDeltaEtaBBarBBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBBarBBar", "#bar{b};#Delta#eta #bar{b};Counts", nBinsEtaB, etaMin, etaMax}, "DeltaEtaBBarQBar");
    auto histDeltaEtaBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBBarLept", "leptons;#Delta#eta #bar{b};Counts", nBinsEta, etaMin, etaMax}, "DeltaEtaBBarLept");


    //------------------------- Delta phi--------------------------------------
    int nBinsPhi = 60;
    int nBinsPhiB = 10;
    double phiMin = -3.14;
    double phiMax = 3.14;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaPhiBQ", "LHEPart_phi[2]-LHEPart_phi[3]").Define("DeltaPhiBQBar", "LHEPart_phi[2]-LHEPart_phi[4]").Define("DeltaPhiBLept", "LHEPart_phi[2]-LHEPart_phi[6]").Define("DeltaPhiBBarQ", "LHEPart_phi[5]-LHEPart_phi[3]").Define("DeltaPhiBBarQBar", "LHEPart_phi[5]-LHEPart_phi[4]").Define("DeltaPhiBBarLept", "LHEPart_phi[5]-LHEPart_phi[6]");

    auto histDeltaPhiBU = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaPhiBU", "u;#Delta#phi b;Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBQ");
    auto histDeltaPhiBC = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBC", "c;#Delta#phi b;Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBQ");
    auto histDeltaPhiBDBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaPhiBDBar", "#bar{d};#Delta#phi b;Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBQBar");
    auto histDeltaPhiBSBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaPhiBSBar", "#bar{s};#Delta#phi b;Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBQBar");
    auto histDeltaPhiBBBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBBBar", "#bar{b};#Delta#phi b;Counts", nBinsPhiB,phiMin,phiMax}, "DeltaPhiBQBar");
    auto histDeltaPhiBLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBLept", "leptons;#Delta#phi b;Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBLept");

    auto histDeltaPhiBBarU = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaPhiBBar", "u;#Delta#phi #bar{b};Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBBarQ");
    auto histDeltaPhiBBarC = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBBarC", "c;#Delta#phi #bar{b};Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBBarQ");
    auto histDeltaPhiBBarDBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaPhiBBarDBar", "#bar{d};#Delta#phi #bar{b};Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBBarQBar");
    auto histDeltaPhiBBarSBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaPhiBBarSBar", "#bar{s};#Delta#phi #bar{b};Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBBarQBar");
    auto histDeltaPhiBBarBBar = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBBarBBar", "#bar{b};#Delta#phi #bar{b};Counts", nBinsPhiB,phiMin,phiMax}, "DeltaPhiBBarQBar");
    auto histDeltaPhiBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBBarLept", "leptons;#Delta#phi #bar{b};Counts", nBinsPhi,phiMin,phiMax}, "DeltaPhiBBarLept");


    //-------------------------------Delta R----------------------------------//
    int nBinsR = 60;
    int nBinsRB = 10;
    double RMin = 0;
    double RMax = 6.5;


    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaRBQ","deltaR(DeltaPhiBQ,DeltaEtaBQ)").Define("DeltaRBQBar","deltaR(DeltaPhiBQBar,DeltaEtaBQBar)").Define("DeltaRBLept","deltaR(DeltaPhiBLept,DeltaEtaBLept)").Define("DeltaRBBarQ","deltaR(DeltaPhiBBarQ,DeltaEtaBBarQ)").Define("DeltaRBBarQBar","deltaR(DeltaPhiBBarQBar,DeltaEtaBBarQBar)").Define("DeltaRBBarLept","deltaR(DeltaPhiBBarLept,DeltaEtaBBarLept)");

    auto histDeltaRBU= ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaRBU", "u;#DeltaR b;Counts", nBinsR,RMin,RMax}, "DeltaRBQ");
    auto histDeltaRBC= ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBC", "c;#DeltaR b;Counts", nBinsR,RMin,RMax}, "DeltaRBQ");
    auto histDeltaRBDBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaRBDBar", "#bar{d};#DeltaR b;Counts", nBinsR,RMin,RMax}, "DeltaRBQBar");
    auto histDeltaRBSBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaRBSBar", "#bar{s};#DeltaR b;Counts", nBinsR,RMin,RMax}, "DeltaRBQBar");
    auto histDeltaRBBBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBBBar", "#bar{b};#DeltaR b;Counts", nBinsRB,RMin,RMax}, "DeltaRBQBar");
    auto histDeltaRBLept= ptEtaPhiMDF.Histo1D({"histDeltaRBLept", "leptons;#DeltaR b;Counts", nBinsR,RMin,RMax}, "DeltaRBLept");

    auto histDeltaRBBarU= ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2").Histo1D({"histDeltaRBBarU", "u;#DeltaR #bar{b};Counts", nBinsR,RMin,RMax}, "DeltaRBBarQ");
    auto histDeltaRBBarC= ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBBarC", "c;#DeltaR #bar{b};Counts", nBinsR,RMin,RMax}, "DeltaRBBarQ");
    auto histDeltaRBBarDBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1").Histo1D({"histDeltaRBBarDBar", "#bar{d};#DeltaR #bar{b};Counts", nBinsR,RMin,RMax}, "DeltaRBBarQBar");
    auto histDeltaRBBarSBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-3").Histo1D({"histDeltaRBBarSBar", "#bar{s};#DeltaR #bar{b};Counts", nBinsR,RMin,RMax}, "DeltaRBBarQBar");
    auto histDeltaRBBarBBar= ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBBarBBar", "#bar{b};#DeltaR #bar{b};Counts", nBinsRB,RMin,RMax}, "DeltaRBBarQBar");
    auto histDeltaRBBarLept= ptEtaPhiMDF.Histo1D({"histDeltaRBarBLept", "leptons;#DeltaR #bar{b};Counts", nBinsR,RMin,RMax}, "DeltaRBBarLept");

    //----------------------------W hadronic decays----------------------------------//

    auto histWPlusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWPlus>0").Histo1D({"histWPlusJetDecay", "W^{+} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWPlus");
    auto histWMinusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWMinus>0").Histo1D({"histWMinusJetDecay", "W^{-} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWMinus");

    StackPlotter jetCouple({histWPlusJetDecay}, "W hadronic Decays", "W qq Decay", "./images/WHadronicDecay.png", false, false, true);

        // Set the TH1 Label of the W decays the strings above
    (jetCouple).SetBinLabel(1, "ud");
    (jetCouple).SetBinLabel(2, "us");
    (jetCouple).SetBinLabel(3, "ub");
    (jetCouple).SetBinLabel(4, "cd");
    (jetCouple).SetBinLabel(5, "cs");
    (jetCouple).SetBinLabel(6, "cb");
    (jetCouple).SetBinLabel(8, "ts");
    (jetCouple).SetBinLabel(9, "tb");

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

    StackPlotter ttbarPt({histPtT, histPtTBar}, "p_{t}(t)/p_{t}(#bar{t})", "p_{t} [GeV]", "./images/pt/PtTTBar.png", true);
    StackPlotter tLeptHadPt({histPtTHad, histPtTLept}, "p_{t}(t#rightarrow q#bar{q})/p_{t}(t#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtTLeptHad.png", true);
    StackPlotter WPMPt({histPtWPlus, histPtWMinus}, "p_{t}(W^{+})/p_{t}(W^{-})", "p_{t} [GeV]", "./images/pt/PtWPlusMinus.png", true);
    StackPlotter WLeptHadPt({histPtWHad, histPtWLept}, "p_{t}(W#rightarrow q#bar{q})/p_{t}(W#rightarrow l#nu)", "p_{t} [GeV]", "./images/pt/PtWLeptHad.png", true);






    StackPlotter etaParticles({histEtaB, histEtaBBar, histEtaC, histEtaCBar, histEtaS, histEtaSBar, histEtaD, histEtaDBar, histEtaU, histEtaUBar, histEtaLept}, "#eta", "#eta", "./images/eta/etaParticles.png");

    etaParticles.SetLegendPos({0.8, 0.6, 0.9, 0.9});
    etaParticles.SetPalette(55);
    etaParticles.SetDrawOpt("hist PMC PLC nostack E0");
    etaParticles.Normalize();

    StackPlotter ptParticles({histPtB, histPtBBar, histPtC, histPtCBar, histPtS, histPtSBar, histPtD, histPtDBar, histPtU, histPtUBar, histPtLept}, "p_{t}", "p_{t} [GeV]", "./images/pt/ptParticles.png");

    ptParticles.SetLegendPos({0.8, 0.6, 0.9, 0.9});
    ptParticles.SetPalette(55);
    ptParticles.SetDrawOpt("hist PMC PLC nostack E0");
    ptParticles.Normalize();


    StackPlotter deltaEtaB({histDeltaEtaBU, histDeltaEtaBC, histDeltaEtaBDBar, histDeltaEtaBSBar, histDeltaEtaBBBar, histDeltaEtaBLept}, "#Delta#eta b", "#Delta#eta", "./images/eta/deltaEtaB.png");
    deltaEtaB.SetPalette(55);
    deltaEtaB.Normalize();
    deltaEtaB.SetDrawOpt("hist PMC PLC nostack E0");
    StackPlotter deltaEtaBBar({histDeltaEtaBBarU, histDeltaEtaBBarC, histDeltaEtaBBarDBar, histDeltaEtaBBarSBar, histDeltaEtaBBarBBar, histDeltaEtaBBarLept}, "#Delta#eta #bar{b}", "#Delta#eta", "./images/eta/deltaEtaBBar.png");
    deltaEtaBBar.SetPalette(55);
    deltaEtaBBar.Normalize();
    deltaEtaBBar.SetDrawOpt("hist PMC PLC nostack E0");
    StackPlotter deltaPhiB({histDeltaPhiBU, histDeltaPhiBC, histDeltaPhiBDBar, histDeltaPhiBSBar, histDeltaPhiBBBar, histDeltaPhiBLept}, "#Delta#phi b", "#Delta#phi", "./images/phi/deltaPhiB.png");
    deltaPhiB.SetPalette(55);
    deltaPhiB.Normalize();
    deltaPhiB.SetDrawOpt("hist PMC PLC nostack E0");
    deltaPhiB.SetLegendPos({0.79, 0.74, 0.92, 0.86});
    StackPlotter deltaPhiBBar({histDeltaPhiBBarU, histDeltaPhiBBarC, histDeltaPhiBBarDBar, histDeltaPhiBBarSBar, histDeltaPhiBBarBBar, histDeltaPhiBBarLept}, "#Delta#phi #bar{b}", "#Delta#phi ", "./images/phi/deltaPhiBBar.png");
    deltaPhiBBar.SetPalette(55);
    deltaPhiBBar.Normalize();
    deltaPhiBBar.SetDrawOpt("hist PMC PLC nostack E0");
    deltaPhiBBar.SetLegendPos({0.79, 0.74, 0.92, 0.86});
    StackPlotter deltaRB({histDeltaRBU, histDeltaRBC, histDeltaRBDBar, histDeltaRBSBar, histDeltaRBBBar, histDeltaRBLept}, "#DeltaR b", "#DeltaR", "./images/r/deltaRB.png");
    deltaRB.SetPalette(55);
    deltaRB.Normalize();
    deltaRB.SetDrawOpt("hist PMC PLC nostack E0");
    StackPlotter deltaRBBar({histDeltaRBBarU, histDeltaRBBarC, histDeltaRBBarDBar, histDeltaRBBarSBar, histDeltaRBBarBBar, histDeltaRBBarLept}, "#DeltaR #bar{b}", "#DeltaR", "./images/r/deltaRBBar.png");
    deltaRBBar.SetPalette(55);
    deltaRBBar.Normalize();
    deltaRBBar.SetDrawOpt("hist PMC PLC nostack E0");


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
        &deltaRBBar
    };

    for (auto v : stackCollection) {
        v->Save();
    }

    //-------------------------------------------------------------------------------
    //                                      DONE
    return 0;
}
