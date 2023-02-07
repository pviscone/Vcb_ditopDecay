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
#include <stdlib.h>

#include "muonUtils.h"

#include "../../utils/itertools/product.hpp"
#include "../../utils/itertools/zip.hpp"

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"

#include "../../utils/DfUtils.h"
#include "../../utils/HistUtils.h"

void muonPreselection(std::string filename, std::string text, std::string imageSaveFolder) {
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";
    datasetText = text;

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();

    ROOT::EnableImplicitMT();
    RDataFrame fileDF("Events", filename,
                      {"LHEPart_pdgId",
                      "nMuon",
                      "Muon_pt",
                      "Muon_eta",
                      "Muon_phi",
                      "Muon_charge",});

    auto muonsDF=fileDF.Filter("selectMuonEvents(LHEPart_pdgId)");

    auto histNmuons = muonsDF.Histo1D({"nMuon", "nMuon", 10, 0, 10}, "nMuon");

    StackPlotter nMuon({histNmuons},"nMuon","",imageSaveFolder);

    std::vector<StackPlotter *> stackCollection {
        &nMuon,
    };

    for (auto v : stackCollection) {
        v->Save();
    }
    exit(EXIT_SUCCESS);
}
