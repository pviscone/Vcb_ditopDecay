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