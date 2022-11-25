#pragma once

#include <Math/Vector4D.h>
#include <TDatabasePDG.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TTree.h>
#include <iostream>

using namespace ROOT::Math;

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

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//                         HISTS
//TODO Generalize this function to accept any number of histograms (vector of histograms)
/**
 * @brief Function that creates, save and return a THStack object of two histograms
 * 
 * @param hist1 First histogram to add to the stack
 * @param hist2 Second histogram to add to the stack
 * @param title Title of the final plot
 * @param xLabel Label of the x axis
 * @param savePath Path/filename.png where the plot will be saved
 * @return THStack* Return the THStack object
 */
THStack *StackHist(TH1 *hist1, TH1 *hist2, std::string title, std::string xLabel, std::string savePath) {
    gStyle->SetPalette(70);
    gStyle->SetOptStat(00011111);

    //! Canvas too big
    THStack *hs = new THStack("hs", title.c_str());
    TCanvas *c = new TCanvas("c", "c", 1000, 750);
    TPad *p1 = new TPad("p1","p1",0.1,0.3,0.9,1.);
    TPad *p2 = new TPad("p2","p2",0.1,0.,0.9,0.34);
    p1->Draw();
    p2->Draw();

    p1->cd();


    hist1->SetMarkerSize(0.3);
    hist2->SetMarkerSize(0.3);

    // Set the range on the y axis to be the same for both histograms (and at 1.1 times the maximum value)
    hist1->GetYaxis()->SetRangeUser(0, 1.15 * std::max(hist1->GetMaximum(), hist2->GetMaximum()));
    hist2->GetYaxis()->SetRangeUser(0, 1.15 * std::max(hist1->GetMaximum(), hist2->GetMaximum()));

    // Add the histograms to the stack
    hs->Add(hist1, "sames");
    hs->Add(hist2, "sames");

    hs->Draw("nostack PLC PMC E1 HIST");
    gPad->Update();

    // Axis lavel
/*     hs->GetXaxis()->SetTitle(xLabel.c_str()); */
    hs->GetYaxis()->SetTitle("Counts");

    //Title
    TLatex T;
    T.SetTextFont(42);
    T.SetTextAlign(21);
    T.DrawLatexNDC(0.55, 0.96, title.c_str());

    //Legend
    TLegend *legend = new TLegend(0.16, 0.85, 0.35, 0.95);
    legend->AddEntry(hist1->GetName(), hist1->GetTitle(), "l");
    legend->AddEntry(hist2->GetName(), hist2->GetTitle(), "l");
    legend->Draw();

    // StatBox
    gPad->Update();
    TPaveStats *st1 = (TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats");
    TPaveStats *st2 = (TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats");
    st1->SetX1NDC(.75);
    st1->SetX2NDC(0.95);
    st1->SetY1NDC(.8);
    st1->SetY2NDC(.92);
    st2->SetX1NDC(.75);
    st2->SetX2NDC(.95);
    st2->SetY1NDC(.66);
    st2->SetY2NDC(.78);
    c->Modified();

    // Ratio plot
    p2->cd();
    TH1F *hist3 = (TH1F *)hist1->Clone("hist3");
    hist3->Divide(hist2);

    hist3->SetStats(0);
    hist3->SetTitle("");
    hist3->GetYaxis()->SetRangeUser(-1,3);
    hist3->GetYaxis()->SetTitle("Ratio");
    hist3->SetLineColor(kBlack);
    hist3->SetMarkerColor(kBlack);
    hist3->SetMarkerSize(0.5);

    hist3->Draw();
    gPad->Update();

    TLine *line = new TLine(p2->GetUxmin(),1.,p2->GetUxmax(),1.);
    line->SetLineColor(kRed);
    line->Draw();
    //Save and return
    c->SaveAs(savePath.c_str());
    return hs;
}
