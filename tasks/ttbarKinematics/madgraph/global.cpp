#pragma once
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

//-------------------------------------------------------------------------------------------------------
//                                 File,tree and branches status
int SetMT() {
    ROOT::EnableImplicitMT();
    return 0;
};
int _ = SetMT();
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
                            .Define("WPlus", PtEtaPhiMVecSum(indexFromWPlus))
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
                       .Define("jetCoupleWMinus", "jetCoupleWMinus(LHEPart_pdgId)")
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

double ptMin = 0;
double ptMax = 500;
int nBinsPt = 60;
int nBinsPtSingle = 60;

double EtaMin = -6;
double EtaMax = 6;
int nBinsEta = 50;
int nBinsEtaSingle = 50;

int nBinsPhi = 60;
int nBinsPhiB = 10;
double phiMin = -3.14;
double phiMax = 3.14;

int nBinsR = 60;
int nBinsRB = 10;
double RMin = 0;
double RMax = 6.5;