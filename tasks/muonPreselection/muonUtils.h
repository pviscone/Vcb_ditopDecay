#pragma once

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"
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


bool selectMuonEvents(const RVec<int> &LHEPart_pdgId){
    if(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13){
        return true;
    }
    else{
        return false;
    }
}

RVec<float> orderAbs(const RVec<float> &Rvec){
    return Reverse(Sort(abs(Rvec)));

}

RVec<float> FourJetsWithoutMuon(const RVec<float> &JetVec,RVec<int> &Jet_muonIdx1){
    RVec<float> RvecWithoutMuon;
    
    bool lessThen5=false;
    int finalSize;
    if (JetVec.size() <= 4 && JetVec.size() > 0) {
        finalSize=JetVec.size();
        lessThen5 = true;
    } else if (JetVec.size() == 0){
        return RvecWithoutMuon;
    } else {
        finalSize = 4;
    }

    int i = 0;
    while(RvecWithoutMuon.size()<=finalSize) {
        if(Jet_muonIdx1[i]!=0){
            RvecWithoutMuon.push_back(JetVec[i]);
        }else{
            if (lessThen5) {
                finalSize = finalSize - 1;
            }
        }
        i++;
    }
    return RvecWithoutMuon;
}
    
