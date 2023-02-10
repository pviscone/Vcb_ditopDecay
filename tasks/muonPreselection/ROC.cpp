#include <ROOT/RDataFrame.hxx>
#include <TMultiGraph.h>
#include <iostream>
using namespace ROOT;

#include "muonUtils.h"

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"

#include "../../utils/DfUtils.h"
#include "../../utils/HistUtils.h"

void ROC(std::string imageSaveFolder) {
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";

    ROOT::EnableImplicitMT();

    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");

    RDataFrame SignalDF("Events", "TTbarSemileptonic_cbOnly_pruned_optimized.root",
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

    auto MuonsFromWDF_signal = SignalDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)");
    auto TriggeredMuonsDF_signal = MuonsFromWDF_signal.Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");
    auto LooseMuonsDF_signal = TriggeredMuonsDF_signal.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("SlimmedJet_pt", "Jet_pt[Jet_jetId>0 && Jet_puId>0]");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("LeadingJetsWithoutMuon_pt", "FourJetsWithoutMuon(SlimmedJet_pt,Jet_muonIdx1)");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_pt>20");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Filter("LeadingJetsWithoutMuon_pt[3]>20");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("Btag_prob", "Jet_btagDeepFlavB[JetMask]");
    LooseMuonsDF_signal = LooseMuonsDF_signal.Define("LeadingJetsWithoutMuon_bTagProb", "FourJetsWithoutMuon(Reverse(Sort(Btag_prob)),Jet_muonIdx1)");

    RDataFrame BackgroundDF("Events", "TTbarSemileptonic_Nocb_optimized.root",
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

    auto MuonsFromWDF_background = BackgroundDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)");
    auto TriggeredMuonsDF_background = MuonsFromWDF_background.Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");
    auto LooseMuonsDF_background = TriggeredMuonsDF_background.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("SlimmedJet_pt", "Jet_pt[Jet_jetId>0 && Jet_puId>0]");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("LeadingJetsWithoutMuon_pt", "FourJetsWithoutMuon(SlimmedJet_pt,Jet_muonIdx1)");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_pt>20");
    LooseMuonsDF_background = LooseMuonsDF_background.Filter("LeadingJetsWithoutMuon_pt[3]>20");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("Btag_prob", "Jet_btagDeepFlavB[JetMask]");
    LooseMuonsDF_background = LooseMuonsDF_background.Define("LeadingJetsWithoutMuon_bTagProb", "FourJetsWithoutMuon(Reverse(Sort(Btag_prob)),Jet_muonIdx1)");

    int N = 100;
    RVec<float> CutsVec = RVec<float>(N + 1);
    int totalEvents_signal = LooseMuonsDF_signal.Count().GetValue();
    int totalEvents_background = LooseMuonsDF_background.Count().GetValue();
    auto cuttedDF_signal = LooseMuonsDF_signal;
    auto cuttedDF_background = LooseMuonsDF_background;

    RVec<float> LeadingBtagAcceptance_signal;
    RVec<float> LeadingBtagAcceptance_background;


    //! THIS IS CATASTROPHICALLY SLOW, OPTIMIZE IT
    for (int i = 1; i <= N; i++) {
        float cut = 1 - pow(10, -2.5 + 2.5 * (float)i / N);
        CutsVec[N-i] = cut;
    }
    CutsVec[N] = 1.;

    std::vector<ROOT::RDF::RResultPtr<double>> count_signal_rr;
    std::vector<ROOT::RDF::RResultPtr<double>> count_background_rr;
    for (int i = 0; i <= N; i++) {
        std::cout << i <<":  Cut: " << CutsVec[i] << std::endl;
        cuttedDF_signal = cuttedDF_signal.Define("BtagMask"+std::to_string(i),"LeadingJetsWithoutMuon_bTagProb[0]>" + std::to_string(CutsVec[i]));
        cuttedDF_background = cuttedDF_background.Define("BtagMask"+std::to_string(i),"LeadingJetsWithoutMuon_bTagProb[0]>" + std::to_string(CutsVec[i]));
        count_signal_rr.push_back(cuttedDF_signal.Sum("BtagMask" + std::to_string(i)));
        count_background_rr.push_back(cuttedDF_background.Sum("BtagMask" + std::to_string(i)));
    }

    for(int i=0; i<=N;i++){

        int count_signal = (int) count_signal_rr[i].GetValue();
        LeadingBtagAcceptance_signal.push_back((float)count_signal / totalEvents_signal);
        int count_background = (int) count_background_rr[i].GetValue();
        LeadingBtagAcceptance_background.push_back((float)count_background / totalEvents_background);
        std::cout << i << ":  Cut: " << CutsVec[i] << "  " << count_signal << std::endl;
    }

    RVec<float> x={0,1};
    RVec<float> y={1,0};
    TGraph diagonal=TGraph(2,x.data(),y.data());
    diagonal.SetLineWidth(2);

    TCanvas *c = new TCanvas("c", "c", 800, 600);
    TGraph graph_signalAcceptance = TGraph (N + 1, CutsVec.data(), LeadingBtagAcceptance_signal.data());
    graph_signalAcceptance.GetXaxis()->SetTitle("Cut on leading btag probability");
    graph_signalAcceptance.GetYaxis()->SetTitle("Signal acceptance");
    graph_signalAcceptance.SetTitle("acceptance vs cut on leading btag probability");
    graph_signalAcceptance.SetLineColor(2);
    graph_signalAcceptance.SetLineWidth(4);
    graph_signalAcceptance.GetYaxis()->SetRangeUser(0., 1);
    graph_signalAcceptance.GetXaxis()->SetRangeUser(0., 1);
    graph_signalAcceptance.Draw("ACP");
    c->SetGrid();
    c->SaveAs((imageSaveFolder + "/LeadingBtagAcceptance_signal.png").c_str());

    TCanvas *c1 = new TCanvas("c1", "c1", 800, 600);
    TGraph graph_backgroundRejection = TGraph(N + 1, CutsVec.data(), (1 - LeadingBtagAcceptance_background).data());
    graph_backgroundRejection.Draw("ACP");
    graph_backgroundRejection.GetXaxis()->SetTitle("Cut on leading btag probability");
    graph_backgroundRejection.GetYaxis()->SetTitle("Background Rejection");
    graph_backgroundRejection.SetTitle("rejection vs cut on leading btag probability");
    graph_backgroundRejection.SetLineColor(2);
    graph_backgroundRejection.SetLineWidth(4);
    graph_backgroundRejection.GetYaxis()->SetRangeUser(0., 1);
    graph_backgroundRejection.GetXaxis()->SetRangeUser(0., 1);
    graph_backgroundRejection.Draw("ACP");
    c1->SetGrid();
    c1->SaveAs((imageSaveFolder + "/LeadingBtagRejection_background.png").c_str());

    TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);
    TGraph graph_ROC =TGraph(N + 1, LeadingBtagAcceptance_signal.data(), (1 - LeadingBtagAcceptance_background).data());
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