#pragma once

#include "CMSStyle/CMS_lumi.C"
#include "CMSStyle/CMS_lumi.h"
#include "CMSStyle/tdrstyle.C"
#include <Math/Vector4D.h>
#include <ROOT/RVec.hxx>
#include <TAxis.h>
#include <TCanvas.h>
#include <TDatabasePDG.h>
#include <TF1.h>
#include <TFrame.h>
#include <THStack.h>
#include <TLeaf.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TTree.h>
#include <iostream>

using namespace ROOT;
using namespace ROOT::Math;
using namespace ROOT::VecOps;



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

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//                         PDG DATABASE

// global pdg database
TDatabasePDG *pdgDatabase = new TDatabasePDG();

/**
 * @brief Function that, given the name of the particle, returns the PDG code
 * If the particle is not found, the function returns 0 (the Rootino)
 *
 * @param name particle name
 * @return int pdgId
 */
int pdg(const char *name) {
    int id;
    if (id == 0) {
        std::cerr << "WARNING: there is a ROOTINO" << std::endl;
    }
    try {
        id = pdgDatabase->GetParticle(name)->PdgCode();
    } catch (std::exception &err) {
        std::cerr << "The particle name: " << name << " does not exist" << std::endl;
        id = 0;
    }
    return id;
}

/**
 * @brief Function that, given the PDG code, returns the name of the particle
 * If the particle does not exist, it returns "Rootino"
 *
 * @param id pdgId
 * @return std::string particle name
 */
std::string pdg(int id) {
    std::string name;
    try {
        name = pdgDatabase->GetParticle(id)->GetName();
    } catch (std::exception &err) {
        std::cerr << "The pdgId: " << id << " does not exist" << std::endl;
        name = "Rootino";
    }
    return name;
}

/**
 * @brief Function that return the TParticlePDG object given the name of the particle
 *
 * @param name particle name
 * @return TParticlePDG*
 */
TParticlePDG *particle(const char *name) {
    return pdgDatabase->GetParticle(name);
}

/**
 * @brief Function that return the TParticlePDG object given the pdgId of the particle
 *
 * @param name particle name
 * @return TParticlePDG*
 */
TParticlePDG *particle(int id) {
    return pdgDatabase->GetParticle(id);
}

bool isQuark(int id) {
    return (abs(id) <= 8);
}

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

float selectQ(const int &WPlusIsHadronic, const RVec<float> &branch) {
    if (WPlusIsHadronic==1) {
        return branch[3];
    } else {
        return branch[6];
    }
}

float selectQBar(const int &WPlusIsHadronic, const RVec<float> &branch) {
    if (WPlusIsHadronic==1) {
        return branch[4];
    } else {
        return branch[7];
    }
}

float selectLept(const int &WPlusIsHadronic, const RVec<float> &branch) {
    if (WPlusIsHadronic==1) {
        return branch[6];
    } else {
        return branch[3];
    }
}

int isWPlusHadronic(const RVec<int> &pdgIdVec) {
    if (TMath::Abs(pdgIdVec[3]) <= 6) {
        return 1;
    }
    else {
        return 0;
    }
}



/**
 * @brief Get the Lorentz Vector object given pt,eta,phi and mass
 *
 * @param tree
 * @param instance
 * @return PtEtaPhiMVector
 */
PtEtaPhiMVector getLorentzVector(TTree *tree, int instance) {
    PtEtaPhiMVector v;
    v.SetPt(tree->GetLeaf("LHEPart_pt")->GetValue(instance));
    v.SetEta(tree->GetLeaf("LHEPart_eta")->GetValue(instance));
    v.SetPhi(tree->GetLeaf("LHEPart_phi")->GetValue(instance));
    v.SetM(tree->GetLeaf("LHEPart_mass")->GetValue(instance));

    return v;
}

PtEtaPhiMVector getLorentzVector(double pt, double eta, double phi, double mass) {
    PtEtaPhiMVector v;
    v.SetPt(pt);
    v.SetEta(eta);
    v.SetPhi(phi);
    v.SetM(mass);

    return v;
}
//-----------------------RDataFrame--------------------------------------
const char *getValue(std::string branch, int idx) {
    std::string branchRow = branch + "[" + std::to_string(idx) + "]";
    char *branchRowChar = new char[branchRow.length() + 1];
    strcpy(branchRowChar, branchRow.c_str());
    return branchRowChar;
}

const char *func(std::string function, std::string arguments) {
    std::string funcArg = function + arguments;
    char *funcArgChar = new char[funcArg.length() + 1];
    strcpy(funcArgChar, funcArg.c_str());
    return funcArgChar;
}

PtEtaPhiMVector PtEtaPhiMVecSum(std::vector<int> idxList, const RVec<float> &ptVec, const RVec<float> &etaVec, const RVec<float> &phiVec, const RVec<float> &mVec, const PtEtaPhiMVector &startVec = {0, 0, 0, 0}) {

    PtEtaPhiMVector vec = startVec;
    for (auto &idx : idxList) {
        float pt = ptVec[idx];
        float eta = etaVec[idx];
        float phi = phiVec[idx];
        float m = mVec[idx];
        vec += PtEtaPhiMVector(pt, eta, phi, m);
    }
    return vec;
}

const char *PtEtaPhiMVecSum(std::vector<int> idxList, std::string ColumnName = "") {
    std::string funcStr = "PtEtaPhiMVecSum({";

    for (auto &idx : idxList) {
        funcStr += std::to_string(idx) + ",";
    }
    funcStr += "},LHEPart_pt,LHEPart_eta,LHEPart_phi,LHEPart_mass";
    if (ColumnName != "") {
        funcStr += "," + ColumnName;
    }
    funcStr += ")";
    char *funcStrChar = new char[funcStr.length() + 1];
    strcpy(funcStrChar, funcStr.c_str());
    return funcStrChar;
}

template <typename T>
RVec<T> filterTracks(const RVec<T> &vec, const RVec<int> mask) {
    RVec<T> filteredVec = vec[mask];
    return filteredVec;
}

double deltaEta(const RVec<float> &etaVec, const int &WPlusIsHadronic, std::string strPart1, std::string strPart2) {
    std::unordered_map<std::string, int> partIdxDictionary;

    partIdxDictionary["B"]=2;
    partIdxDictionary["Bbar"]=5;
    if(WPlusIsHadronic==1){
        partIdxDictionary["Q"]=3;
        partIdxDictionary["Qbar"]=4;
        partIdxDictionary["Lept"]=6;
    } else {
        partIdxDictionary["Q"]=6;
        partIdxDictionary["Qbar"]=7;
        partIdxDictionary["Lept"]=3;
    }
    return etaVec[partIdxDictionary[strPart1]] - etaVec[partIdxDictionary[strPart2]];
}


double deltaR(double deltaPhi, double deltaEta) {
    return TMath::Sqrt(TMath::Power(deltaPhi, 2) + TMath::Power(deltaEta, 2));
}

double deltaPhi(const RVec<float> &phiVec, const int &WPlusIsHadronic, std::string strPart1, std::string strPart2) {
    std::unordered_map<std::string, int> partIdxDictionary;

    partIdxDictionary["B"] = 2;
    partIdxDictionary["Bbar"] = 5;
    if (WPlusIsHadronic == 1) {
        partIdxDictionary["Q"] = 3;
        partIdxDictionary["Qbar"] = 4;
        partIdxDictionary["Lept"] = 6;
    } else {
        partIdxDictionary["Q"] = 6;
        partIdxDictionary["Qbar"] = 7;
        partIdxDictionary["Lept"] = 3;
    }
    double dphi = phiVec[partIdxDictionary[strPart1]] - phiVec[partIdxDictionary[strPart2]];
    if (dphi > TMath::Pi()) {
        dphi -= 2 * TMath::Pi();
    } else if (dphi < -TMath::Pi()) {
        dphi += 2 * TMath::Pi();
    }
    return dphi;
}


RVec<float> leading(const RVec<float> &vec, const int &WPlusIsHadronic, bool absoluteValue=false) {
    RVec<float> quarkVec;
    if (WPlusIsHadronic==1){
        quarkVec = {vec[2],vec[3],vec[4],vec[5]};
    } else {
        quarkVec = {vec[2],vec[6],vec[7],vec[5]};
    }
    if (absoluteValue) {
        quarkVec=abs(quarkVec);
    }
    auto sortQuarkVec = Sort(quarkVec);
    return Reverse(sortQuarkVec);
}

RVec<float> leadingIdx(const RVec<int> &pdgIdVec,const RVec<float> &vec,const int &WPlusIsHadronic,bool absoluteValue=false) {
    RVec<int> quarkOrdered {0,0,0,0};
    RVec<float> quarkVec;
    if (WPlusIsHadronic==1) {
        quarkVec = {vec[2], vec[3], vec[4], vec[5]};
    } else {
        quarkVec = {vec[2], vec[6], vec[7], vec[5]};
    }
    if (absoluteValue) {
        quarkVec=abs(quarkVec);
    }
    for(int i = 0; i < quarkVec.size(); i++){
        int argMax = ArgMax(quarkVec);
        quarkVec[argMax] = -999999999;
        quarkOrdered[i]=argMax+2;
    }
    return quarkOrdered;

}


RVec<float> orderAccordingToVec(const RVec<float> &vecToOrder, const RVec<float> &orderVec, const int &WPlusIsHadronic, bool absoluteValue=false) {
    RVec<float> partVec;
    RVec<float> newVecToOrder;
    if(WPlusIsHadronic==1){
        partVec={orderVec[2], orderVec[3], orderVec[4], orderVec[5]};
        newVecToOrder={vecToOrder[2], vecToOrder[3], vecToOrder[4], vecToOrder[5]};
    } else {
        partVec={orderVec[2], orderVec[6], orderVec[7], orderVec[5]};
        newVecToOrder={vecToOrder[2], vecToOrder[6], vecToOrder[7], vecToOrder[5]};
    }
    RVec<float> resultVec{0, 0, 0, 0};
    if (absoluteValue) {
        partVec = abs(partVec);
        newVecToOrder = abs(newVecToOrder);
    }
    for (int i = 0; i < partVec.size(); i++) {
        int argMax = ArgMax(partVec);
        partVec[argMax] = std::numeric_limits<float>::lowest();
        resultVec[i] = newVecToOrder[argMax];
    }
    return resultVec;
}

//first element is the pdgid the smallest element, second element is the smallest element
float Min(float r1, float r2, float r3, float r4){
    return std::min({r1,r2,r3,r4});
}

//1:b, 2:q, 3:qbar, 4:bbar, 5:lept
int MaskMin(const float &r1, const float &r2, const float &r3, const float &r4,const std::string &part1){
    std::map<std::string, int> partMaskDictionary = {{"B",1},{"Q",2},{"Qbar",3},{"Bbar",4},{"Lept",5}};
    partMaskDictionary.erase(part1);

    std::vector<int> maskVec;
    for(auto const &mapElement : partMaskDictionary){
        maskVec.push_back(mapElement.second);

    }
    std::sort(maskVec.begin(), maskVec.end());
    RVec<float> vec = {r1,r2,r3,r4};
    int argmin = ArgMin(vec);
    return maskVec[argmin];
}

double ARGMIN(const RVec<double> &v) {
    return ArgMin(v);
};
