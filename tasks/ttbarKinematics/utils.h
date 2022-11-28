#pragma once

#include <Math/Vector4D.h>
#include <TDatabasePDG.h>
#include <TFrame.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TTree.h>
#include <iostream>
#include <TAxis.h>
#include <TF1.h>

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
// TODO Generalize this function to accept any number of histograms (vector of histograms)
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
THStack *StackHist(TH1 *hist1, TH1 *hist2, std::string title, std::string xLabel, std::string savePath,bool fit = false) {
    // gStyle->SetPalette(70);

    // Set the colors,size and the stat options
    gStyle->SetOptStat(00011111);

    int hist1Color = 798;
    int hist2Color = 920;

    hist1->SetFillColorAlpha(hist1Color, 1);
    hist1->SetLineColor(hist1Color);
    hist1->SetMarkerColor(1);
    hist1->SetMarkerSize(0.);

    hist2->SetFillColorAlpha(hist2Color, 0.3);
    hist2->SetLineColor(1);
    hist2->SetLineWidth(1);
    hist2->SetMarkerColor(1);
    hist2->SetMarkerSize(0.5);

    // Create the stack
    THStack *hs = new THStack("hs", title.c_str());

    TCanvas *c = new TCanvas("cName", "cName", 50, 50, 800, 700);

    // Create 2 pads
    TPad *p1 = new TPad("pad1", "pad1", 0, 0.3, 0.95, 0.98);
    TPad *p2 = new TPad("pad2", "pad2", 0, 0.05, 0.95, 0.3);
    p1->Draw();
    p2->Draw();

    // First pad: THStack
    p1->cd();
    p1->SetBottomMargin(0.); // Upper and lower plot are joined

    // Set the range on the y axis to be the same for both histograms (and at 1.1 times the maximum value)
    hist1->GetYaxis()->SetRangeUser(0, 1.15 * std::max(hist1->GetMaximum(), hist2->GetMaximum()));
    hist2->GetYaxis()->SetRangeUser(0, 1.15 * std::max(hist1->GetMaximum(), hist2->GetMaximum()));

    // Add the histograms to the stack
    hs->Add(hist1, "sames");
    hs->Add(hist2, "sames");

    hs->Draw("nostack HIST");
    gPad->Update();

    // Thing to avoid the overlap on the first label
    hist1->GetYaxis()->SetLabelSize(0.);
    TAxis *axis = hs->GetYaxis();
    axis->ChangeLabel(1, -1, -1, -1, -1, -1, " ");
    axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)
    axis->SetLabelSize(15);

    // THStack axis
    /*     hs->GetXaxis()->SetTitle(xLabel.c_str()); */
    hs->GetYaxis()->SetTitle("Events");
    hs->GetYaxis()->SetLabelOffset(0.01);

    // Title
    TLatex TeX;
    TeX.SetTextFont(42);
    TeX.SetTextAlign(21);
    TeX.DrawLatexNDC(0.55, 0.97, title.c_str());

    // Legend
    TLegend *legend = new TLegend(0.75, 0.51, 0.95, 0.68);
    legend->AddEntry(hist1->GetName(), hist1->GetTitle(), "f");
    legend->AddEntry(hist2->GetName(), hist2->GetTitle(), "f");

    // FIT
    if(fit){
        p1->cd();
        TF1 *f1 = new TF1("f1", "breitwigner", hist1->GetXaxis()->GetXmin(), hist1->GetXaxis()->GetXmax());
        TF1 *f2 = new TF1("f2", "breitwigner", hist2->GetXaxis()->GetXmin(), hist2->GetXaxis()->GetXmax());

        f1->SetParameters(hist1->GetMaximum(), hist1->GetMean(), hist1->GetRMS());
        f2->SetParameters(hist2->GetMaximum(), hist2->GetMean(), hist2->GetRMS());

        hist1->Fit("f1","L 0");
        hist2->Fit("f2","L 0");

        f1->SetNpx(500);
        f1->SetLineColor(2);
        f1->SetLineWidth(1);

        f2->SetNpx(500);
        f2->SetLineColor(9);
        f2->SetLineWidth(1);

        f1->Draw("same");
        f2->Draw("same");

        std::string leg1=hist1->GetTitle();
        leg1+="\\mbox{ BW fit}";
        std::string leg2 = hist2->GetTitle();
        leg2 += "\\mbox{ BW fit}";

        legend->AddEntry("f1", leg1.c_str(), "l");
        legend->AddEntry("f2", leg2.c_str(), "l");
    }


    legend->SetBorderSize(0);
    legend->Draw();

    // StatBox
    gPad->Update();
    TPaveStats *st1 = (TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats");
    TPaveStats *st2 = (TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats");
    st1->SetX1NDC(.75);
    st1->SetX2NDC(0.95);
    st1->SetY1NDC(.7);
    st1->SetY2NDC(.92);
    st2->SetX1NDC(.53);
    st2->SetX2NDC(.73);
    st2->SetY1NDC(.7);
    st2->SetY2NDC(.92);

    st1->SetLineColor(hist1Color);
    st2->SetLineColor(hist2Color);

    c->Modified();

    // Ratio plot
    p2->cd();
    p2->SetTopMargin(0);
    p2->SetBottomMargin(0.2);
    TH1F *hist3 = (TH1F *)hist1->Clone("hist3");
    hist3->Divide(hist2);

    // Ratio plot options
    hist3->SetStats(0);
    hist3->SetTitle("");
    hist3->SetLineColor(kBlack);
    hist3->SetMarkerColor(kBlack);
    hist3->SetMarkerSize(0.5);
    hist3->Draw();
    gPad->Update();

    // Y axis ratio plot settings
    hist3->GetYaxis()->SetRangeUser(-1.2, 3.2);
    hist3->GetYaxis()->SetTitle("Ratio");
    hist3->GetYaxis()->SetNdivisions(505);
    hist3->GetYaxis()->SetTitleSize(20);
    hist3->GetYaxis()->SetTitleFont(43);
    hist3->GetYaxis()->SetTitleOffset(1.55);
    hist3->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
    hist3->GetYaxis()->SetLabelSize(15);

    // X axis ratio plot settings
    hist3->GetXaxis()->SetTitleSize(20);
    hist3->GetXaxis()->SetTitleFont(43);
    hist3->GetXaxis()->SetTitleOffset(0.7);
    hist3->GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
    hist3->GetXaxis()->SetLabelSize(15);
    gPad->Modified();
    gPad->Update();
    c->RedrawAxis();

    // Red line on ratio=1
    TLine *line = new TLine(p2->GetUxmin(), 1., p2->GetUxmax(), 1.);
    line->SetLineColor(kRed);
    line->Draw();

    // Draw CMS on top left
    int iPos = 11;
    float iPeriod = 0;
    CMS_lumi(c, iPeriod, iPos);

    // Save and return
    c->Update();
    c->RedrawAxis();
    c->GetFrame()->Draw();
    c->SaveAs(savePath.c_str());

    return hs;
}
