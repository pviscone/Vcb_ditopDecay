#pragma once

#include "../../../utils/CMSStyle/CMS_lumi.C"
#include "../../../utils/CMSStyle/CMS_lumi.h"
#include "../../../utils/CMSStyle/tdrstyle.C"
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

double deltaR(double deltaPhi, double deltaEta) {
    return TMath::Sqrt(TMath::Power(deltaPhi, 2) + TMath::Power(deltaEta, 2));
}

double deltaPhi(double phi1, double phi2){
    double dphi = phi1 - phi2;
    if (dphi > TMath::Pi()) {
        dphi -= 2 * TMath::Pi();
    } else if (dphi < -TMath::Pi()) {
        dphi += 2 * TMath::Pi();
    }
    return dphi;
}

RVec<float> leading(const RVec<float> &vec, bool absoluteValue=false) {
    RVec<float> quarkVec {vec[2],vec[3],vec[4],vec[5],vec[6]};
    if (absoluteValue) {
        quarkVec=abs(quarkVec);
    }
    auto sortQuarkVec = Sort(quarkVec);
    return Reverse(sortQuarkVec);
}

RVec<float> leadingIdx(const RVec<int> &pdgIdVec,const RVec<float> &vec,bool absoluteValue=false) {
    RVec<int> quarkOrdered {0,0,0,0,0};
    RVec<float> quarkVec {vec[2],vec[3],vec[4],vec[5],vec[6]};
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

RVec<float> quarkVec(const RVec<int> &pdgIdVec) {
    return RVec<float> {(float) pdgIdVec[3],(float) -pdgIdVec[4]};
}

RVec<float> orderAccordingToVec(const RVec<float> &vecToOrder, const RVec<float> &orderVec, bool absoluteValue=false) {
    RVec<float> partVec{orderVec[2], orderVec[3], orderVec[4], orderVec[5], orderVec[6]};
    RVec<float> newVecToOrder{vecToOrder[2], vecToOrder[3], vecToOrder[4], vecToOrder[5], vecToOrder[6]};
    RVec<float> resultVec{0, 0, 0, 0, 0};
    if (absoluteValue) {
        partVec = abs(partVec);
        newVecToOrder = abs(newVecToOrder);
    }
    for (int i = 0; i < partVec.size(); i++) {
        int argMax = ArgMax(partVec);
        partVec[argMax] = -999999999;
        resultVec[i] = newVecToOrder[argMax];
    }
    return resultVec;
}

//first element is the pdgid the smallest element, second element is the smallest element
float DeltaRMin(float r1, float r2, float r3, float r4){
    return std::min({r1,r2,r3,r4});
}

float PartDeltaRMin(float r1, float r2, float r3, float r4,float toExclude){
    RVec<float> vec {r1, r2, r3, r4 };
    float argmin = ArgMin(vec)+2;
    if (argmin >= toExclude){
        argmin++;
    }
    return argmin;
}
