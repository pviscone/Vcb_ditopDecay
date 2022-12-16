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
using namespace ROOT::RDF;







    int stackPlotterCounter = 0;
class StackPlotter {
private:
    std::vector<RResultPtr<::TH1D>> RResultVector;
    Int_t palette=-1;
    int nStatBox = 2;
    int maxStatBoxPrinted;
    int iPos = 11;
    float statGap = 0.18;
    std::vector<float> legendPos{0.79, 0.62, 0.92, 0.74};
    std::vector<float> statPos{.78, .76, .95, .92};
    float yAxisMultiplier = 1.3;
    std::string drawOpt = "nostack hist";
    TCanvas *c;
    std::vector<TH1 *> histVector;
    THStack *hs;
    std::string title;
    std::string xLabel;
    bool ratio = false;
    bool log = false;
    bool fit = false;
    bool normalize = false;
    bool statsInLegend=true;
    std::string savePath;
    bool constructed = false;
    std::string canvasName = "cName";
    std::string colors;


public:
    std::vector<int> histColor;
    std::vector<double> alphaColor;
    std::vector<int> lineColor;
    std::vector<int> markerColor;
    std::vector<double> markerSize;
    std::vector<int> fitColor;
    std::vector<double> lineAlpha;
    int lineWidth;
    int fitWidth;
    void SetColors(std::string colors) {
        if (colors == "YellowBlack") {
            histColor = {798, 920,863,616,418};
            alphaColor = {0.8, 0.4,0.3,0.25,0.25};
            lineColor = {798, 920+2, 863, 616,418};
            lineAlpha = {1, 0.7,0.7,0.7,0.7};
            markerColor = {1, 1,1,1,1};
            markerSize = {0, 0.5,0,0,0};
            fitColor = {2, 9, 417,0,0};
            fitWidth = 2;
            lineWidth = 3;
        } else {
            std::cout << "ERROR: colors not implemented" << std::endl;
            std::exit(1);
        }
        this->colors = colors;
    };
    StackPlotter(std::vector<RResultPtr<::TH1D>> RResultVector, std::string title, std::string xLabel, std::string savePath = "", bool ratio = false, bool fit = false, bool log = false) {
        hs = new THStack("hs", title.c_str());
        canvasName += std::to_string(stackPlotterCounter);
        stackPlotterCounter++;
        this->RResultVector = RResultVector;
        this->savePath = savePath;
        this->title = title;
        this->xLabel = xLabel;
        this->ratio = ratio;
        this->fit = fit;
        this->log = log;
        colors="YellowBlack";
        this->SetColors(colors);
        maxStatBoxPrinted = nStatBox;
    };
    ~StackPlotter(){};
    void Normalize() {
        this->normalize = true;
    };
    void SetIPos(int iPos) {
        this->iPos = iPos;
    };
    void SetMaxStatBoxPrinted(int maxStatBoxPrinted) {
        if(maxStatBoxPrinted>nStatBox){
            throw std::invalid_argument("maxStatBoxPrinted cannot be larger than nStatBox");
        }
        this->maxStatBoxPrinted = maxStatBoxPrinted;
    };
    void SetLegendPos(std::vector<float> legendPos) {
        this->legendPos = legendPos;
    };
    void SetLegendPos(int idx, float pos) {
        this->legendPos[idx] = pos;
    };
    void SetStatPos(std::vector<float> statPos) {
        this->statPos = statPos;
    };
    void SetYAxisMultiplier(float yAxisMultiplier) {
        this->yAxisMultiplier = yAxisMultiplier;
    };
    void SetStatsInLegend(bool val){
        this->statsInLegend=val;
    }
    void SetStatPos(int idx, float pos) {
        this->statPos[idx] = pos;
    };
    void SetStatGap(float statGap) {
        this->statGap = statGap;
    };
    void SetSavePath(std::string savePath) {
        this->savePath = savePath;
    };
    void SetLog() {
        this->log = true;
    };
    void SetFit(bool fit) {
        this->fit = fit;
    };
    void SetRatio(bool ratio) {
        this->ratio = ratio;
    };
    void SetLineWidth(int lineWidth) {
        this->lineWidth = lineWidth;
    };
    void SetNStatBox(int nStatBox) {
        this->nStatBox = nStatBox;
    };
    void SetFitWidth(int fitWidth) {
        this->fitWidth = fitWidth;
    };
    void SetDrawOpt(std::string drawOpt) {
        this->drawOpt = drawOpt;
    };
    void AddDrawOpt(std::string drawOpt) {
        this->drawOpt += " " + drawOpt;
    };
    void SetPalette(Int_t palette) {
        this->palette = palette;
        gStyle->SetPalette(palette);
        this->AddDrawOpt("PLC");
    };
    void SetPalette(Int_t palette,Float_t alpha) {
        this->palette = palette;
        gStyle->SetPalette(palette,alpha);
        this->AddDrawOpt("PLC");
    };
    THStack *GetStack() {
        return hs;
    };
    TCanvas *GetCanvas() {
        return c;
    };
    void Add(TH1* hist){
        histVector.push_back(hist);
    }
    void GetValue() {
        c = new TCanvas(canvasName.c_str(), canvasName.c_str(), 50, 50, 1280, 1120);
        
        for (auto &resultptr : RResultVector) {
            TH1D hist = resultptr.GetValue();
            TH1D *histptr = new TH1D;
            *histptr = hist;
            histVector.push_back(histptr);
        }
        constructed = true;
    }
    void SetBinLabel(int idx, std::string label) {
        if (!constructed) {
            GetValue();
        }
        for (auto &&hist : histVector) {
            hist->GetXaxis()->SetBinLabel(idx, label.c_str());
        }
    }


    void Draw() {

        if (!constructed) {
            GetValue();
        }
        c->cd();

        if (histVector.size() > nStatBox) {
            gStyle->SetOptStat(00000000);
            legendPos={0.7, 0.7, 0.92, 0.92};
        } else {
            gStyle->SetOptStat(00011110);
        }

        TPad *p1;
        TPad *p2;
        std::vector<TPaveStats *> statsVector(nStatBox);
        std::vector<TF1 *> tfVector(histVector.size());
        std::string padName = "pad";
        padName += stackPlotterCounter;
        if (ratio) {
            if (histVector.size() != 2) {
                std::cerr << "ERROR: Ratio plot can be done only with two histograms" << std::endl;
                ratio = false;
            } else {
                p1 = new TPad((padName + "_1").c_str(), (padName + "_1").c_str(), 0, 0.3, 0.95, 0.98);
                p2 = new TPad((padName + "_2").c_str(), (padName + "_2").c_str(), 0, 0.05, 0.95, 0.3);
                p1->Draw();
                p2->Draw();

                // First pad: THStack
                p1->cd();
                p1->SetBottomMargin(0.); // Upper and lower plot are joined
            }
        } else {
            p1 = new TPad((padName + "_1").c_str(), (padName + "_1").c_str(), 0, 0.00, 0.95, 1);
            p2 = nullptr;
            p1->Draw();
            p1->cd();
        }
        TLegend *legend = new TLegend(legendPos[0], legendPos[1], legendPos[2], legendPos[3]);
        legend->SetFillColorAlpha(0,0.4);
        p1->cd();
        for (int idx = 0; idx < histVector.size(); idx++) {
            if(palette<0){
                histVector[idx]->SetFillColorAlpha(histColor[idx], alphaColor[idx]);
                histVector[idx]->SetLineColorAlpha(lineColor[idx],lineAlpha[idx]);
                histVector[idx]->SetMarkerColor(markerColor[idx]);
                histVector[idx]->SetMarkerSize(markerSize[idx]);
            }
            histVector[idx]->SetLineWidth(lineWidth);

            if (normalize) {
                histVector[idx]->Scale(1. / histVector[idx]->Integral(), "width");
            }
            if (!log) {
                histVector[idx]->GetYaxis()->SetRangeUser(0, yAxisMultiplier * std::max(histVector[idx]->GetMaximum(), histVector[idx]->GetMaximum()));
            } else {
                histVector[idx]->GetYaxis()->SetRangeUser(0.1, yAxisMultiplier * std::max(histVector[idx]->GetMaximum(), 11.5 * histVector[idx]->GetMaximum()));
                gPad->SetLogy();
            }

            hs->Add(histVector[idx], "sames");
            hs->Draw(drawOpt.c_str());
            gPad->Update();

            std::string legendLabel=histVector[idx]->GetTitle();
            if(histVector.size()>nStatBox && statsInLegend==true){
                legendLabel+="  ";
                legendLabel += std::to_string(histVector[idx]->GetMean()).substr(0, std::to_string(histVector[idx]->GetMean()).find(".") + 3);
                legendLabel += " (";
                legendLabel += std::to_string(histVector[idx]->GetRMS()).substr(0, std::to_string(histVector[idx]->GetRMS()).find(".") + 3);
                legendLabel += ") ";
                try{
                    legendLabel += xLabel.substr(xLabel.find("["),xLabel.find("]"));
                }catch(const std::exception& e){}
            }
            std::string legendOption = "f";
            if(palette>0){
                legendOption = "l";
            }
            legend->AddEntry(histVector[idx]->GetName(), legendLabel.c_str(), legendOption.c_str());

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
                if(idx<maxStatBoxPrinted){
                    statsVector[idx]->SetX1NDC(statPos[0] - idx * statGap);
                    statsVector[idx]->SetY1NDC(statPos[1]);
                    statsVector[idx]->SetX2NDC(statPos[2] - idx * statGap);

                    statsVector[idx]->SetY2NDC(statPos[3]);
                } else{
                    statsVector[idx]->SetX1NDC(99999);
                    statsVector[idx]->SetY1NDC(99999);
                    statsVector[idx]->SetX2NDC(99999);
                    statsVector[idx]->SetY2NDC(99999);
                }

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
        hs->GetYaxis()->SetMaxDigits(3);
        // Title
        TLatex TeX;
        TeX.SetTextFont(42);
        TeX.SetTextSize(0.036);
        TeX.SetTextAlign(21);
        TeX.DrawLatexNDC(0.55, 0.97, title.c_str());

        // Legend
        legend->SetBorderSize(0);
        legend->Draw();

        // Draw CMS on top left

        float iPeriod = 0;
        CMS_lumi(c, iPeriod, iPos);

        //---------------------------RATIO PLOT--------------------------------
        if (ratio) {
            // Thing to avoid the overlap on the first label
            histVector[0]->GetYaxis()->SetLabelSize(0.);
            TAxis *axis = hs->GetYaxis();
            axis->ChangeLabel(1, -1, -1, -1, -1, -1, " ");
            axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)
            axis->SetLabelSize(25);

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
            histRatio->GetYaxis()->SetRangeUser(0.3, 1.7);
            histRatio->GetYaxis()->SetTitle("Ratio");
            histRatio->GetYaxis()->SetNdivisions(505);
            histRatio->GetYaxis()->SetTitleSize(32);
            histRatio->GetYaxis()->SetTitleFont(43);
            histRatio->GetYaxis()->SetTitleOffset(1.55);
            histRatio->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
            histRatio->GetYaxis()->SetLabelSize(25);

            // X axis ratio plot settings
            histRatio->GetXaxis()->SetTitleSize(30);
            histRatio->GetXaxis()->SetTitleFont(43);
            histRatio->GetXaxis()->SetTitleOffset(0.8);
            histRatio->GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
            histRatio->GetXaxis()->SetLabelSize(25);
            gPad->Modified();
            gPad->Update();
            c->RedrawAxis();

            // Red line on ratio=1
            TLine *line = new TLine(p2->GetUxmin(), 1., p2->GetUxmax(), 1.);
            line->SetLineColor(kRed);
            line->Draw();
            if (log) {
                histVector[0]->GetYaxis()->SetRangeUser(1, yAxisMultiplier*10 * std::max(histVector[0]->GetMaximum(), histVector[1]->GetMaximum()));
                p1->SetLogy();
            }
        } else {
            hs->GetXaxis()->SetTitle(xLabel.c_str());
        }
        // Save and return
        gPad->Modified();
        c->Update();
        c->RedrawAxis();
        //c->GetFrame()->Draw();
    };

    void Save(std::string path) {
        Draw();
        c->SaveAs(path.c_str());
    }
    void Save() {
        Draw();
        c->SaveAs(savePath.c_str());
    }
    void setPdgLabel() {
        this->SetBinLabel(1, "d");
        this->SetBinLabel(2, "u");
        this->SetBinLabel(3, "s");
        this->SetBinLabel(4, "c");
        this->SetBinLabel(5, "b");
        this->SetBinLabel(6, "t");
    }
    void setQuarkTypeLabel() {
        this->SetBinLabel(1, "q up family");
        this->SetBinLabel(2, "q down family");
    }
};

