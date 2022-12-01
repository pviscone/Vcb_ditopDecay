#pragma once

#include <Math/Vector4D.h>
#include <TAxis.h>
#include <TDatabasePDG.h>
#include <TF1.h>
#include <TFrame.h>
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
// TODO Generalize this function to accept any number of histograms (vector of histograms)
/**
 * @brief Function that creates, save and return a THStack object of two histograms
 *
 * @param histVector Vector of histograms to add to the stack
 * @param title Title of the final plot
 * @param xLabel Label of the x axis
 * @param savePath Path/filename.png where the plot will be saved
 * @param ratio If true, the plot will have a ratio plot
 * @param fit If true, the plot will have a Breit-Wigner fit
 * @param log If true, the plot will be in log YScale
 * @return THStack* Return the THStack object
 */
THStack *StackHist(std::vector<TH1 *> &histVector, std::string title, std::string xLabel, std::string savePath, bool ratio = false, bool fit = false, bool log = false) {

    //-------------------------DECLARATIONS--------------------------------
    // Set the colors,size and the stat options
    gStyle->SetOptStat(00011110);

    // TODO Add other colors
    std::vector<int> histColor{798, 920};
    std::vector<double> alphaColor{1., 0.3};
    std::vector<int> lineColor{798, 1};
    std::vector<int> markerColor{1, 1};
    std::vector<double> markerSize{0, 0.5};
    std::vector<int> fitColor{
        2,
        9,
    };
    int fitWidth = 2;

    int nStatBox = 3;

    std::string drawHSOption = "nostack hist";

    TCanvas *c = new TCanvas("cName", "cName", 50, 50, 800, 700);
    THStack *hs = new THStack("hs", title.c_str());

    TPad *p1;
    TPad *p2;
    std::vector<TPaveStats *> statsVector(nStatBox);
    std::vector<TF1 *> tfVector(histVector.size());

    if (ratio) {
        if (histVector.size() != 2) {
            std::cerr << "ERROR: Ratio plot can be done only with two histograms" << std::endl;
            ratio = false;
        } else {
            p1 = new TPad("pad1", "pad1", 0, 0.3, 0.95, 0.98);
            p2 = new TPad("pad2", "pad2", 0, 0.05, 0.95, 0.3);
            p1->Draw();
            p2->Draw();

            // First pad: THStack
            p1->cd();
            p1->SetBottomMargin(0.); // Upper and lower plot are joined
        }
    }
    if (!ratio) {
        p1 = new TPad("pad1", "pad1", 0, 0.00, 0.95, 1);
        p2 = nullptr;
        p1->Draw();
        p1->cd();
    }
    TLegend *legend = new TLegend(0.79, 0.62, 0.92, 0.74);

    // Set the range on the y axis to be the same for both histograms (and at 1.1 times the maximum value)
    histVector[0]->GetYaxis()->SetRangeUser(0, 1.15 * std::max(histVector[0]->GetMaximum(), histVector[1]->GetMaximum()));
    histVector[1]->GetYaxis()->SetRangeUser(0, 1.15 * std::max(histVector[0]->GetMaximum(), histVector[1]->GetMaximum()));

    //---------------------LOOP OVER HISTOGRAMS--------------------------------
    p1->cd();
    for (int idx = 0; idx < histVector.size(); idx++) {

        if (log) {
            histVector[idx]->GetYaxis()->SetRangeUser(0.1, 1.15 * std::max(histVector[idx]->GetMaximum(), 11.5 * histVector[idx]->GetMaximum()));
            gPad->SetLogy();
        }

        histVector[idx]->SetFillColorAlpha(histColor[idx], alphaColor[idx]);
        histVector[idx]->SetLineColor(lineColor[idx]);
        histVector[idx]->SetMarkerColor(markerColor[idx]);
        histVector[idx]->SetMarkerSize(markerSize[idx]);
        hs->Add(histVector[idx], "sames");

        hs->Draw(drawHSOption.c_str());
        gPad->Update();

        legend->AddEntry(histVector[idx]->GetName(), histVector[idx]->GetTitle(), "f");

        if (fit) {
            std::string funcName = "f" + std::to_string(idx);
            TF1 *f = new TF1(funcName.c_str(), "breitwigner", histVector[idx]->GetXaxis()->GetXmin(), histVector[idx]->GetXaxis()->GetXmax());

            f->SetParameters(histVector[idx]->GetMaximum(), histVector[idx]->GetMean(), histVector[idx]->GetRMS());
            histVector[idx]->Fit(funcName.c_str(), "L 0");

            f->SetNpx(500);
            f->SetLineColor(fitColor[idx]);
            f->SetLineWidth(fitWidth);

            tfVector[idx] = f;
        }

        if (histVector.size() <= nStatBox) {
            statsVector[idx] = (TPaveStats *)histVector[idx]->GetListOfFunctions()->FindObject("stats");
            statsVector[idx]->SetX1NDC(.78 - idx * 0.18);
            statsVector[idx]->SetX2NDC(0.95 - idx * 0.18);
            statsVector[idx]->SetY1NDC(.78);
            statsVector[idx]->SetY2NDC(.92);
            statsVector[idx]->SetLineColor(histColor[idx]);
            c->Modified();
        }
    }

    gPad->Update();
    if (fit) {
        int idx = 0;
        for (auto &&func : tfVector) {
            func->Draw("same");
            std::string funcName = "f" + std::to_string(idx);
            std::string legFit = histVector[idx]->GetTitle();
            legFit += " BW fit";
            legend->AddEntry(funcName.c_str(), legFit.c_str(), "l");
            idx++;
        }
    }

    hs->GetYaxis()->SetTitle("Events");
    hs->GetYaxis()->SetLabelOffset(0.01);

    // Title
    TLatex TeX;
    TeX.SetTextFont(42);
    TeX.SetTextSize(0.038);
    TeX.SetTextAlign(21);
    TeX.DrawLatexNDC(0.55, 0.97, title.c_str());

    // Legend
    legend->SetBorderSize(0);
    legend->Draw();

    // Draw CMS on top left
    int iPos = 11;
    float iPeriod = 0;
    CMS_lumi(c, iPeriod, iPos);

    //---------------------------RATIO PLOT--------------------------------
    if (ratio) {
        // Thing to avoid the overlap on the first label
        histVector[0]->GetYaxis()->SetLabelSize(0.);
        TAxis *axis = hs->GetYaxis();
        axis->ChangeLabel(1, -1, -1, -1, -1, -1, " ");
        axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)
        axis->SetLabelSize(15);

        // Ratio plot
        p2->cd();
        p2->SetTopMargin(0);
        p2->SetBottomMargin(0.2);
        TH1F *histRatio = (TH1F *)histVector[0]->Clone("histRatio");
        histRatio->Divide(histVector[1]);

        // Ratio plot options
        histRatio->SetStats(0);
        histRatio->SetTitle("");
        histRatio->SetLineColor(kBlack);
        histRatio->SetMarkerColor(kBlack);
        histRatio->SetMarkerSize(0.5);
        histRatio->Draw();
        gPad->Update();

        // Y axis ratio plot settings
        histRatio->GetYaxis()->SetRangeUser(-1.2, 3.2);
        histRatio->GetYaxis()->SetTitle("Ratio");
        histRatio->GetYaxis()->SetNdivisions(505);
        histRatio->GetYaxis()->SetTitleSize(20);
        histRatio->GetYaxis()->SetTitleFont(43);
        histRatio->GetYaxis()->SetTitleOffset(1.55);
        histRatio->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
        histRatio->GetYaxis()->SetLabelSize(15);

        // X axis ratio plot settings
        histRatio->GetXaxis()->SetTitleSize(20);
        histRatio->GetXaxis()->SetTitleFont(43);
        histRatio->GetXaxis()->SetTitleOffset(0.75);
        histRatio->GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
        histRatio->GetXaxis()->SetLabelSize(15);
        gPad->Modified();
        gPad->Update();
        c->RedrawAxis();

        // Red line on ratio=1
        TLine *line = new TLine(p2->GetUxmin(), 1., p2->GetUxmax(), 1.);
        line->SetLineColor(kRed);
        line->Draw();
        if (log) {
            histVector[0]->GetYaxis()->SetRangeUser(0.1, 1.15 * std::max(histVector[0]->GetMaximum(), histVector[1]->GetMaximum()));
            p1->SetLogy();
        }
    } else {
        hs->GetXaxis()->SetTitle(xLabel.c_str());
    }

    // Save and return
    gPad->Modified();
    c->Update();
    c->RedrawAxis();
    c->GetFrame()->Draw();
    c->SaveAs(savePath.c_str());

    return hs;
}
