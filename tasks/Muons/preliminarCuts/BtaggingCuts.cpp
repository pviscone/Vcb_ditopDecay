#include <ROOT/RDataFrame.hxx>
#include <TMultiGraph.h>
#include <iostream>
#include <regex>
using namespace ROOT;



#include "../../../utils/CMSStyle/CMS_lumi.C"
#include "../../../utils/CMSStyle/CMS_lumi.h"
#include "../../../utils/CMSStyle/tdrstyle.C"

#include "../../../utils/DfUtils.h"
#include "../../../utils/HistUtils.h"

void BtaggingCuts(std::string imageSaveFolder) {
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";

    ROOT::EnableImplicitMT();

    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");

    RDataFrame SignalDF("Events", "../TTbarSemileptonic_cbOnly_pruned_optimized.root",
                        {"LHEPart_pdgId",
                         "nMuon",
                         "Muon_pt",
                         "Muon_eta",
                         "Muon_phi",
                         "Muon_charge",
                         "Muon_looseId",
                         "Muon_mediumId",
                         "Muon_tightId",
                         "Muon_pfIsoId",
                         "Jet_jetId",
                         "Jet_puId",
                         "Jet_pt",
                         "Jet_muonIdx1",
                         "Jet_btagDeepFlavB"});

    auto LooseMuonsDF_signal = SignalDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)").Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

    LooseMuonsDF_signal = LooseMuonsDF_signal.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_muonIdx1!=0 && Jet_pt>20");
    for (auto &name : LooseMuonsDF_signal.GetColumnNames()) {
        if (regex_match(name, std::regex("Jet_.*"))) {
            LooseMuonsDF_signal = LooseMuonsDF_signal.Redefine(name, name + "[JetMask]");
        }
    }
    LooseMuonsDF_signal = LooseMuonsDF_signal.Redefine("nJet", "Jet_pt.size()");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Filter("nJet>=4");
    //Btagging
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("LeadingJetsWithoutMuon_bTagProb", "Reverse(Sort(Jet_btagDeepFlavB))");

    RDataFrame BackgroundDF("Events", "../TTbarSemileptonic_Nocb.root",
                            {"LHEPart_pdgId",
                             "nMuon",
                             "Muon_pt",
                             "Muon_eta",
                             "Muon_phi",
                             "Muon_charge",
                             "Muon_looseId",
                             "Muon_mediumId",
                             "Muon_tightId",
                             "Muon_pfIsoId",
                             "Jet_jetId",
                             "Jet_puId",
                             "Jet_pt",
                             "Jet_muonIdx1",
                             "Jet_btagDeepFlavB"});

    auto LooseMuonsDF_background = BackgroundDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)").Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

    LooseMuonsDF_background = LooseMuonsDF_background.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_muonIdx1!=0 && Jet_pt>20");
    for (auto &name : LooseMuonsDF_background.GetColumnNames()) {
        if (regex_match(name, std::regex("Jet_.*"))) {
            LooseMuonsDF_background = LooseMuonsDF_background.Redefine(name, name + "[JetMask]");
        }
    }
    LooseMuonsDF_background = LooseMuonsDF_background.Redefine("nJet", "Jet_pt.size()");
    LooseMuonsDF_background = LooseMuonsDF_background.Filter("nJet>=4");
    // Btagging
    LooseMuonsDF_background = LooseMuonsDF_background.Define("LeadingJetsWithoutMuon_bTagProb", "Reverse(Sort(Jet_btagDeepFlavB))");

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    int N = 100;
    RVec<float> CutsVec = RVec<float>(N + 1);
    int totalEvents_signal = LooseMuonsDF_signal.Count().GetValue();
    int totalEvents_background = LooseMuonsDF_background.Count().GetValue();
    auto cuttedDF_signal = LooseMuonsDF_signal;
    auto cuttedDF_background = LooseMuonsDF_background;

    RVec<float> LeadingBtagAcceptance_signal;
    RVec<float> LeadingBtagAcceptance_background;

    RVec<float> SecondBtagAcceptance_signal;
    RVec<float> SecondBtagAcceptance_background;


    for (int i = 1; i <= N; i++) {
        float cut = 1 - pow(10, -1 + 1 * (float)i / N);
        CutsVec[N-i] = cut;
    }
    CutsVec[N] = 1.;

    std::vector<ROOT::RDF::RResultPtr<double>> count_signal_rr;
    std::vector<ROOT::RDF::RResultPtr<double>> count_background_rr;
    std::vector<ROOT::RDF::RResultPtr<double>> count_signal_rrSecond;
    std::vector<ROOT::RDF::RResultPtr<double>> count_background_rrSecond;
    // Medium cut on btagDeepFlavB
    auto cuttedDF_signalSecond = cuttedDF_signal.Filter("LeadingJetsWithoutMuon_bTagProb[0]>0.2783");
    auto cuttedDF_backgroundSecond = cuttedDF_background.Filter("LeadingJetsWithoutMuon_bTagProb[0]>0.2783");
    for (int i = 0; i <= N; i++) {
        cuttedDF_signal = cuttedDF_signal.Define("BtagMask"+std::to_string(i),"LeadingJetsWithoutMuon_bTagProb[0]>" + std::to_string(CutsVec[i]));
        cuttedDF_background = cuttedDF_background.Define("BtagMask"+std::to_string(i),"LeadingJetsWithoutMuon_bTagProb[0]>" + std::to_string(CutsVec[i]));
        count_signal_rr.push_back(cuttedDF_signal.Sum("BtagMask" + std::to_string(i)));
        count_background_rr.push_back(cuttedDF_background.Sum("BtagMask" + std::to_string(i)));

        //Second btag: cut on leading btag
        cuttedDF_signalSecond = cuttedDF_signalSecond.Define("BtagMaskSecond" + std::to_string(i), "LeadingJetsWithoutMuon_bTagProb[1]>" + std::to_string(CutsVec[i]));
        cuttedDF_backgroundSecond = cuttedDF_backgroundSecond.Define("BtagMaskSecond" + std::to_string(i), "LeadingJetsWithoutMuon_bTagProb[1]>" + std::to_string(CutsVec[i]));
        count_signal_rrSecond.push_back(cuttedDF_signalSecond.Sum("BtagMaskSecond" + std::to_string(i)));
        count_background_rrSecond.push_back(cuttedDF_backgroundSecond.Sum("BtagMaskSecond" + std::to_string(i)));
    }

    int totalEvents_signalSecond = cuttedDF_signalSecond.Count().GetValue();
    int totalEvents_backgroundSecond = cuttedDF_backgroundSecond.Count().GetValue();

    for(int i=0; i<=N;i++){

        int count_signal = (int) count_signal_rr[i].GetValue();
        LeadingBtagAcceptance_signal.push_back((float)count_signal / totalEvents_signal);
        int count_background = (int) count_background_rr[i].GetValue();
        LeadingBtagAcceptance_background.push_back((float)count_background / totalEvents_background);

        int count_signalSecond = (int)count_signal_rrSecond[i].GetValue();
        SecondBtagAcceptance_signal.push_back((float)count_signalSecond / totalEvents_signalSecond);
        int count_backgroundSecond = (int)count_background_rrSecond[i].GetValue();
        SecondBtagAcceptance_background.push_back((float)count_backgroundSecond/ totalEvents_backgroundSecond);
    }


    TCanvas *c = new TCanvas("c", "c", 800, 600);
    TMultiGraph mg;
    TGraph graph_signalAcceptance = TGraph (N + 1, CutsVec.data(), LeadingBtagAcceptance_signal.data());
    graph_signalAcceptance.SetLineColor(2);
    graph_signalAcceptance.SetLineWidth(4);
    graph_signalAcceptance.SetName("signalLeading");

    TGraph graph_backgroundAcceptance = TGraph(N + 1, CutsVec.data(), (LeadingBtagAcceptance_background).data());
    graph_backgroundAcceptance.SetLineColor(4);
    graph_backgroundAcceptance.SetLineWidth(4);
    graph_backgroundAcceptance.SetName("backgroundLeading");



    mg.GetYaxis()->SetTitle("acceptance");
    mg.GetXaxis()->SetTitle("cut on btagDeepFlavB");
    mg.SetTitle("btag acceptance");

    mg.GetYaxis()->SetRangeUser(0.9, 1);
    mg.GetXaxis()->SetRangeUser(0., 0.4);
    mg.Add(&graph_signalAcceptance);
    mg.Add(&graph_backgroundAcceptance);

    mg.Draw("ACP");

    auto legend = new TLegend(0.7, 0.7, 0.95, 0.9);
    legend->AddEntry(&graph_signalAcceptance, "signal Leading", "l");
    legend->AddEntry(&graph_backgroundAcceptance, "background Leading", "l");


    legend->Draw();


    c->SetGrid();
    c->SaveAs((imageSaveFolder + "/LeadingBtagAcceptance.png").c_str());

    TCanvas *c1 = new TCanvas("c1", "c1", 800, 600);
    TMultiGraph mg1;
    TGraph graph_backgroundAcceptance2 = TGraph(N + 1, CutsVec.data(), (SecondBtagAcceptance_background).data());
    graph_backgroundAcceptance2.SetLineColor(6);
    graph_backgroundAcceptance2.SetLineWidth(4);
    graph_backgroundAcceptance2.SetName("backgroundSecond");

    TGraph graph_signalAcceptance2 = TGraph(N + 1, CutsVec.data(), SecondBtagAcceptance_signal.data());
    graph_signalAcceptance2.SetLineColor(7);
    graph_signalAcceptance2.SetLineWidth(4);
    graph_signalAcceptance2.SetName("signalSecond");
    mg1.Add(&graph_backgroundAcceptance2);
    mg1.Add(&graph_signalAcceptance2);
    mg1.GetYaxis()->SetTitle("acceptance");
    mg1.GetXaxis()->SetTitle("cut on btagDeepFlavB");
    mg1.SetTitle("btag acceptance");

    mg1.GetYaxis()->SetRangeUser(0.7, 1.01);
    mg1.GetXaxis()->SetRangeUser(0., 0.15);
    mg1.Draw("ACP");
    auto legend2 = new TLegend(0.7, 0.7, 0.95, 0.9);
    legend2->SetHeader("Cut on jet btag leading probability>0.2783","C");
    legend2->AddEntry(&graph_signalAcceptance2, "signal Second", "l");
    legend2->AddEntry(&graph_backgroundAcceptance2, "background Second", "l");
    legend2->Draw();
    c1->SetGrid();
    c1->SaveAs((imageSaveFolder + "/SecondBtagAcceptance.png").c_str());

    RVec<float> x = {0, 1};
    RVec<float> y = {1, 0};
    TGraph diagonal = TGraph(2, x.data(), y.data());
    diagonal.SetLineWidth(2);
    TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);
    TGraph graph_ROC = TGraph(N + 1, LeadingBtagAcceptance_signal.data(), (1 - LeadingBtagAcceptance_background).data());
    TMultiGraph mg3;
    graph_ROC.SetLineColor(2);
    graph_ROC.SetLineWidth(4);
    mg3.GetXaxis()->SetTitle("Signal efficiency");
    mg3.GetYaxis()->SetTitle("Background rejection");
    mg3.SetTitle("ROC curve");
    mg3.GetYaxis()->SetRangeUser(0., 1);
    mg3.GetXaxis()->SetRangeUser(0., 1);
    mg3.Add(&graph_ROC);
    mg3.Add(&diagonal);
    mg3.Draw("ACP");
    c2->SetGrid();
    c2->SaveAs((imageSaveFolder + "/LeadingBtagAcceptance_ROC.png").c_str());
}
