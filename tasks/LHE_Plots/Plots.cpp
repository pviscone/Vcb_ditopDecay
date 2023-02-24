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


#include "../../utils/itertools/product.hpp"
#include "../../utils/itertools/zip.hpp"

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"

#include "../../utils/DfUtils.h"
#include "../../utils/HistUtils.h"


void Plots(std::string rootFile, std::string datasetName, std::string imageSaveFolder) {
    ROOT::EnableImplicitMT();
    RDataFrame fileDF("Events", rootFile,
                  {"nLHEPart",
                   "LHEPart_pt",
                   "LHEPart_eta",
                   "LHEPart_phi",
                   "LHEPart_mass",
                   "LHEPart_pdgId"});

    int nEvents = fileDF.Count().GetValue();

    // If you want to run on 10 event (for debugging purpose), uncomment, disable the MT and rename fileDF to fileDF10
    // auto fileDF=fileDF0.Range(10000);

    //-------------------------------------------------------------------------------------------------------
    //      Define a Lorentz vectors for each particles

    auto lorentzVectorsDF = fileDF
                                .Define("WMinus", PtEtaPhiMVecSum(indexFromWMinus))
                                .Define("WPlus", PtEtaPhiMVecSum(indexFromWPlus))
                                .Define("T", PtEtaPhiMVecSum({indexQFromT}, "WPlus"))
                                .Define("TBar", PtEtaPhiMVecSum({indexQBarFromTBar}, "WMinus"));

    auto ptEtaPhiMDF = lorentzVectorsDF
                        .Define("WMinus_pt", "WMinus.pt()")
                        .Define("WMinus_eta", "WMinus.eta()")
                        .Define("WMinus_phi", "WMinus.phi()")
                        .Define("WMinus_mass", "WMinus.mass()")
                        .Define("WPlus_pt", "WPlus.pt()")
                        .Define("WPlus_eta", "WPlus.eta()")
                        .Define("WPlus_phi", "WPlus.phi()")
                        .Define("WPlus_mass", "WPlus.mass()")
                        .Define("T_pt", "T.pt()")
                        .Define("T_eta", "T.eta()")
                        .Define("T_phi", "T.phi()")
                        .Define("T_mass", "T.mass()")
                        .Define("TBar_pt", "TBar.pt()")
                        .Define("TBar_eta", "TBar.eta()")
                        .Define("TBar_phi", "TBar.phi()")
                        .Define("TBar_mass", "TBar.mass()")
                        .Define("jetCoupleWPlus", "jetCoupleWPlus(LHEPart_pdgId)")
                        .Define("jetCoupleWMinus", "jetCoupleWMinus(LHEPart_pdgId)")
                        .Define("WLept_mass", "isLeptonic(jetCoupleWPlus,WPlus_mass,WMinus_mass)")
                        .Define("WLept_pt", "isLeptonic(jetCoupleWPlus,WPlus_pt,WMinus_pt)")
                        .Define("WLept_eta", "isLeptonic(jetCoupleWPlus,WPlus_eta,WMinus_eta)")
                        .Define("WLept_phi", "isLeptonic(jetCoupleWPlus,WPlus_phi,WMinus_phi)")
                        .Define("WHad_mass", "isHadronic(jetCoupleWPlus,WPlus_mass,WMinus_mass)")
                        .Define("WHad_pt", "isHadronic(jetCoupleWPlus,WPlus_pt,WMinus_pt)")
                        .Define("WHad_eta", "isHadronic(jetCoupleWPlus,WPlus_eta,WMinus_eta)")
                        .Define("WHad_phi", "isHadronic(jetCoupleWPlus,WPlus_phi,WMinus_phi)")
                        .Define("TLept_mass", "isLeptonic(jetCoupleWPlus,T_mass,TBar_mass)")
                        .Define("TLept_pt", "isLeptonic(jetCoupleWPlus,T_pt,TBar_pt)")
                        .Define("TLept_eta", "isLeptonic(jetCoupleWPlus,T_eta,TBar_eta)")
                        .Define("TLept_phi", "isLeptonic(jetCoupleWPlus,T_phi,TBar_phi)")
                        .Define("THad_mass", "isHadronic(jetCoupleWPlus,T_mass,TBar_mass)")
                        .Define("THad_pt", "isHadronic(jetCoupleWPlus,T_pt,TBar_pt)")
                        .Define("THad_eta", "isHadronic(jetCoupleWPlus,T_eta,TBar_eta)")
                        .Define("THad_phi", "isHadronic(jetCoupleWPlus,T_phi,TBar_phi)")
                        .Define("isWPlusHadronic", isWPlusHadronic,{"LHEPart_pdgId"})
                        .Define("jetCouple",jetCouple,{"jetCoupleWPlus","jetCoupleWMinus"})
                        .Define("Lepton",Lept,{"LHEPart_pdgId"});

    double ptMin = 0;
    double ptMax = 500;
    int nBinsPt = 60;
    int nBinsPtSingle = 60;

    double EtaMin = -6;
    double EtaMax = 6;
    int nBinsEta = 50;
    int nBinsEtaSingle = 50;

    int nBinsPhi = 60;
    int nBinsPhiB = 10;
    double phiMin = -3.14;
    double phiMax = 3.14;

    int nBinsR = 60;
    int nBinsRB = 10;
    double RMin = 0;
    double RMax = 6.5;


#pragma region Header
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";
    datasetText = datasetName;

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();


#pragma endregion Header



//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma region RECO

    double massW = 80.385;
    double massTop = 172.5;
    double widthW = 2.085;
    double widthTop = 1.41;
    double plotWidthMultiplier = 6.;

    double wideMultiplier = 4.;
    double binWideMultiplier = 2.;

    double massWmin = massW - plotWidthMultiplier * widthW;
    double massWmax = massW + plotWidthMultiplier * widthW;
    double massTopmin = massTop - plotWidthMultiplier * widthTop;
    double massTopmax = massTop + plotWidthMultiplier * widthTop;

    double massWminWide = massW - plotWidthMultiplier * widthW * wideMultiplier;
    double massWmaxWide = massW + plotWidthMultiplier * widthW * wideMultiplier;
    double massTopminWide = massTop - plotWidthMultiplier * widthTop * wideMultiplier;
    double massTopmaxWide = massTop + plotWidthMultiplier * widthTop * wideMultiplier;

    int nBinsTop = 4 * (2 * plotWidthMultiplier * widthTop);
    int nBinsW = 4 * (2 * plotWidthMultiplier * widthW);
    int nBinsTopWide = nBinsTop * binWideMultiplier;
    int nBinsWWide = nBinsW * binWideMultiplier;

    // non si sa perchè ma getptr è rottissimpo. usa getvalue e poi passa gli indirizzi

    auto histMTLept = ptEtaPhiMDF.Histo1D({"histMTLept", "t#rightarrow bl#nu;M_{t} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TLept_mass");
    auto histMTHad = ptEtaPhiMDF.Histo1D({"histMTHad", "t#rightarrow bq#bar{q};M_{t}  [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "THad_mass");

    auto histMT = ptEtaPhiMDF.Histo1D({"histMT", "t;M_{t} [GeV]; Counts", nBinsTop, massTopmin, massTopmax}, "T_mass");
    auto histMTBar = ptEtaPhiMDF.Histo1D({"histMTBar", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTop, massTopmin, massTopmax}, "TBar_mass");

    auto histMTLeptWide = ptEtaPhiMDF.Histo1D({"histMTLeptWide", "t#rightarrow l#nu;M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TLept_mass");
    auto histMTHadWide = ptEtaPhiMDF.Histo1D({"histMTHadWide", "t#rightarrow q#bar{q};M_{t}  [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "THad_mass");

    auto histMTWide = ptEtaPhiMDF.Histo1D({"histMTWide", "t;M_{t} [GeV]; Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "T_mass");
    auto histMTBarWide = ptEtaPhiMDF.Histo1D({"histMTBarWide", "#bar{t}; M_{#bar{t}} [GeV];Counts", nBinsTopWide, massTopminWide, massTopmaxWide}, "TBar_mass");

    auto histMWLept = ptEtaPhiMDF.Histo1D({"histMWLept", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WLept_mass");
    auto histMWHad = ptEtaPhiMDF.Histo1D({"histMWHad", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsW, massWmin, massWmax}, "WHad_mass");

    auto histMWPlus = ptEtaPhiMDF.Histo1D({"histMWPlus", "W^{+};M_{W^{+}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WPlus_mass");
    auto histMWMinus = ptEtaPhiMDF.Histo1D({"histMWMinus", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsW, massWmin, massWmax}, "WMinus_mass");

    auto histMWLeptWide = ptEtaPhiMDF.Histo1D({"histMWLeptWide", "W#rightarrow l#nu;M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WLept_mass");
    auto histMWHadWide = ptEtaPhiMDF.Histo1D({"histMWHadWide", "W#rightarrow q#bar{q};M_{W} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WHad_mass");

    auto histMWPlusWide = ptEtaPhiMDF.Histo1D({"histMWPlusWide", "W^{+};M_{W^{+}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WPlus_mass");
    auto histMWMinusWide = ptEtaPhiMDF.Histo1D({"histMWMinusWide", "W^{-}; M_{W^{-}} [GeV];Counts", nBinsWWide, massWminWide, massWmaxWide}, "WMinus_mass");

    //-------------------------------------Pt-------------------------------------//


    auto histPtTLept = ptEtaPhiMDF.Histo1D({"histPtTLept", "t#rightarrow bl#nu;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TLept_pt");
    auto histPtTHad = ptEtaPhiMDF.Histo1D({"histPtTHad", "t#rightarrow bq#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "THad_pt");

    auto histPtT = ptEtaPhiMDF.Histo1D({"histPtT", "t; p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "T_pt");
    auto histPtTBar = ptEtaPhiMDF.Histo1D({"histPtTBar", "#bar{t} ;p_{#bar{t}} [GeV];Counts", nBinsPt, ptMin, ptMax}, "TBar_pt");

    auto histPtWLept = ptEtaPhiMDF.Histo1D({"histPtWLept", "W#rightarrow l#nu;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WLept_pt");
    auto histPtWHad = ptEtaPhiMDF.Histo1D({"histPtWHad", "W#rightarrow q#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WHad_pt");

    auto histPtWPlus = ptEtaPhiMDF.Histo1D({"histPtWPlus", "W^{+};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WPlus_pt");
    auto histPtWMinus = ptEtaPhiMDF.Histo1D({"histPtWMinus", "W^{-};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "WMinus_pt");

    // -------------------------------------Eta-------------------------------------//


    auto histEtaTLept = ptEtaPhiMDF.Histo1D({"histEtaTLept", "t#rightarrow bl#nu;#eta;Counts", nBinsEta, EtaMin, EtaMax}, "TLept_eta");
    auto histEtaTHad = ptEtaPhiMDF.Histo1D({"histEtaTHad", "t#rightarrow bq#bar{q};#eta;Counts", nBinsEta, EtaMin, EtaMax}, "THad_eta");

    auto histEtaT = ptEtaPhiMDF.Histo1D({"histEtaT", "t;#eta;Counts", nBinsEta, EtaMin, EtaMax}, "T_eta");
    auto histEtaTBar = ptEtaPhiMDF.Histo1D({"histEtaTBar", "#bar{t};#eta;Counts", nBinsEta, EtaMin, EtaMax}, "TBar_eta");

    auto histEtaWLept = ptEtaPhiMDF.Histo1D({"histEtaWLept", "W#rightarrow l#nu;#eta;Counts", nBinsEta, EtaMin, EtaMax}, "WLept_eta");
    auto histEtaWHad = ptEtaPhiMDF.Histo1D({"histEtaWHad", "W#rightarrow q#bar{q};#eta;Counts", nBinsEta, EtaMin, EtaMax}, "WHad_eta");

    auto histEtaWPlus = ptEtaPhiMDF.Histo1D({"histEtaWPlus", "W^{+};#eta;Counts", nBinsEta, EtaMin, EtaMax}, "WPlus_eta");
    auto histEtaWMinus = ptEtaPhiMDF.Histo1D({"histEtaWMinus", "W^{-};#eta;Counts", nBinsEta, EtaMin, EtaMax}, "WMinus_eta");

    //----------------------------W hadronic decays----------------------------------//

    auto histWPlusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWPlus>0").Histo1D({"histWPlusJetDecay", "W^{+} q#bar{q} decay; ;Counts", 9, 1, 9}, "jetCoupleWPlus");
    auto histWMinusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWMinus>0").Histo1D({"histWMinusJetDecay", "W^{-} q#bar{q} decay; ;Counts", 9, 1, 9}, "jetCoupleWMinus");

    StackPlotter jetCouple({histWPlusJetDecay,histWMinusJetDecay}, "W hadronic Decays", "", imageSaveFolder+"/WHadronicDecay.png", false, false, true);

    jetCouple.SetDrawOpt("hist");
    jetCouple.SetLegendPos({0.78, 0.55, 0.95, 0.7});

    jetCouple.SetStatsInLegend(false);



    // Set the TH1 Label of the W decays the strings above
    (jetCouple).SetBinLabel(1, "ud");
    (jetCouple).SetBinLabel(2, "us");
    (jetCouple).SetBinLabel(3, "ub");
    (jetCouple).SetBinLabel(4, "cd");
    (jetCouple).SetBinLabel(5, "cs");
    (jetCouple).SetBinLabel(6, "cb");
    (jetCouple).SetBinLabel(7, "td");
    (jetCouple).SetBinLabel(8, "ts");
    (jetCouple).SetBinLabel(9, "tb");

    //jetCouple.SetMaxStatBoxPrinted(1);



    if(datasetName.find("cb") == string::npos){
        TH1D *histCKM = new TH1D("CKM", "CKM Expected", 9, 1, 9);

        histCKM->SetBinContent(jetCoupleDictionary["ud"], nEvents * TMath::Power(CKM["ud"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["us"], nEvents * TMath::Power(CKM["us"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["ub"], nEvents * TMath::Power(CKM["ub"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["cd"], nEvents * TMath::Power(CKM["cd"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["cs"], nEvents * TMath::Power(CKM["cs"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["cb"], nEvents * TMath::Power(CKM["cb"], 2) / 2);
        histCKM->SetBinContent(jetCoupleDictionary["td"], 0);
        histCKM->SetBinContent(jetCoupleDictionary["ts"], 0);
        histCKM->SetBinContent(jetCoupleDictionary["tb"], 0);
        jetCouple.Add(histCKM);
    }

    StackPlotter jetCoupleNormalized({histWPlusJetDecay,histWMinusJetDecay}, "W hadronic Decays", "", imageSaveFolder+"/WHadronicDecayNormalized.png", false, false, false);

    jetCoupleNormalized.SetDrawOpt("hist");
    jetCoupleNormalized.SetLegendPos({0.78, 0.55, 0.95, 0.7});

    jetCoupleNormalized.SetStatsInLegend(false);

    // Set the TH1 Label of the W decays the strings above
    (jetCoupleNormalized).SetBinLabel(1, "ud");
    (jetCoupleNormalized).SetBinLabel(2, "us");
    (jetCoupleNormalized).SetBinLabel(3, "ub");
    (jetCoupleNormalized).SetBinLabel(4, "cd");
    (jetCoupleNormalized).SetBinLabel(5, "cs");
    (jetCoupleNormalized).SetBinLabel(6, "cb");
    (jetCoupleNormalized).SetBinLabel(7, "td");
    (jetCoupleNormalized).SetBinLabel(8, "ts");
    (jetCoupleNormalized).SetBinLabel(9, "tb");
    jetCoupleNormalized.SetYLabel("Fraction");
    jetCoupleNormalized.Normalize("stack binwise");




    auto histLeptE = ptEtaPhiMDF.Filter("Lepton==11").Histo1D({"histLeptE", "e; ;Counts", 9, 1, 9}, "jetCouple");
    auto histLeptMu = ptEtaPhiMDF.Filter("Lepton==13").Histo1D({"histLeptMu", "#mu; ;Counts", 9, 1, 9}, "jetCouple");
    auto histLeptTau = ptEtaPhiMDF.Filter("Lepton==15").Histo1D({"histLeptETau", "#tau; ;Counts", 9, 1, 9}, "jetCouple");


    StackPlotter leptons({histLeptE,histLeptMu,histLeptTau}, "Leptons", "", imageSaveFolder+"/Leptons.png", false, false, false);

    leptons.Normalize("stack binwise");
    leptons.SetDrawOpt("hist");
    leptons.SetLegendPos({0.78, 0.55, 0.95, 0.7});

    leptons.SetStatsInLegend(false);

    (leptons).SetBinLabel(1, "ud");
    (leptons).SetBinLabel(2, "us");
    (leptons).SetBinLabel(3, "ub");
    (leptons).SetBinLabel(4, "cd");
    (leptons).SetBinLabel(5, "cs");
    (leptons).SetBinLabel(6, "cb");
    (leptons).SetBinLabel(7, "td");
    (leptons).SetBinLabel(8, "ts");
    (leptons).SetBinLabel(9, "tb");
    leptons.SetYLabel("Fraction");







    StackPlotter ttbarMass({histMTBar, histMT}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", imageSaveFolder+"/mass/Mttbar.png", true, true, false);
    StackPlotter tLeptHadMass({histMTHad, histMTLept}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", imageSaveFolder+"/mass/MtLeptHad.png", true, true);
    StackPlotter WPMMass({histMWPlus, histMWMinus}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", imageSaveFolder+"/mass/MWPlusMinus.png", true, true);
    StackPlotter WLeptHadMass({histMWHad, histMWLept}, "M_{W#rightarrow q#bar{q} }/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", imageSaveFolder+"/mass/MWLeptHad.png", true, true);

    StackPlotter ttbarMassWide({histMTWide, histMTBarWide}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", imageSaveFolder+"/mass/MttbarWide.png", true);
    StackPlotter tLeptHadMassWide({histMTHadWide, histMTLeptWide}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", imageSaveFolder+"/mass/MtLeptHadWide.png", true);
    StackPlotter WPMMassWide({histMWPlusWide, histMWMinusWide}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", imageSaveFolder+"/mass/MWPlusMinusWide.png", true);
    StackPlotter WLeptHadMassWide({histMWHadWide, histMWLeptWide}, "M_{W#rightarrow q#bar{q}}/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", imageSaveFolder+"/mass/MWLeptHadWide.png", true);

    StackPlotter ttbarEta({histEtaT, histEtaTBar}, "#eta_{t}/#eta_{#bar{t}}", "#eta_{t}", imageSaveFolder+"/eta/EtaTTbar.png", true);
    StackPlotter tLeptHadEta({histEtaTHad, histEtaTLept}, "#eta_{t#rightarrow q#bar{q}} / #eta_{t#rightarrow l#nu}", "#eta_{t}", imageSaveFolder+"/eta/EtaTLeptHad.png", true);
    StackPlotter WPMEta({histEtaWPlus, histEtaWMinus}, "#eta_{W^{+}}/#eta_{W^{-}}", "#eta_{W}", imageSaveFolder+"/eta/EtaWPlusMinux.png", true);
    StackPlotter WLeptHadEta({histEtaWHad, histEtaWLept}, "#eta_{W#rightarrow q#bar{q}}/#eta_{W#rightarrow l#nu}", "#eta_{W}", imageSaveFolder+"/eta/EtaWLeptHad.png", true);

    StackPlotter ttbarPt({histPtT, histPtTBar}, "p_{T}(t)/p_{T}(#bar{t})", "p_{T} [GeV]", imageSaveFolder+"/pt/PtTTBar.png", true);
    StackPlotter tLeptHadPt({histPtTHad, histPtTLept}, "p_{T}(t#rightarrow q#bar{q})/p_{T}(t#rightarrow l#nu)", "p_{T} [GeV]", imageSaveFolder+"/pt/PtTLeptHad.png", true);
    StackPlotter WPMPt({histPtWPlus, histPtWMinus}, "p_{T}(W^{+})/p_{T}(W^{-})", "p_{T} [GeV]", imageSaveFolder+"/pt/PtWPlusMinus.png", true);
    StackPlotter WLeptHadPt({histPtWHad, histPtWLept}, "p_{T}(W#rightarrow q#bar{q})/p_{T}(W#rightarrow l#nu)", "p_{T} [GeV]", imageSaveFolder+"/pt/PtWLeptHad.png", true);

    ttbarMass.SetNStatBox(2);
    tLeptHadMass.SetNStatBox(2);
    WPMMass.SetNStatBox(2);
    WLeptHadMass.SetNStatBox(2);

#pragma endregion RECO



//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------










#pragma region Particles

#pragma region PT(Particles)
    auto histPtB = ptEtaPhiMDF.Define("B_pt", "LHEPart_pt[2]").Histo1D({"histPtB", "b;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "B_pt");
    auto histPtBBar = ptEtaPhiMDF.Define("BBar_pt", "LHEPart_pt[5]").Histo1D({"histPtBBar", "#bar{b};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "BBar_pt");
    auto histPtQ = ptEtaPhiMDF.Define("Q_pt", selectQ,{"isWPlusHadronic","LHEPart_pt"}).Histo1D({"histPtQ", "q;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Q_pt");
    auto histPtQBar = ptEtaPhiMDF.Define("QBar_pt", selectQBar,{"isWPlusHadronic","LHEPart_pt"}).Histo1D({"histPtQBar", "#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "QBar_pt");
    auto histPtLept = ptEtaPhiMDF.Define("Lept_pt",selectLept,{"isWPlusHadronic","LHEPart_pt"}).Histo1D({"histPtL", "l;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Lept_pt");

    StackPlotter ptParticles({histPtB, histPtBBar, histPtQ, histPtQBar, histPtLept}, "p_{T}", "p_{T} [GeV]", imageSaveFolder+"/pt/ptParticles.png");

    ptParticles.SetLegendPos({0.7, 0.6, 0.9, 0.9});
    ptParticles.SetPalette(55);
    ptParticles.SetDrawOpt("hist PMC PLC nostack");
    ptParticles.SetLineWidth(3);

#pragma endregion PT(Particles)
#pragma region ETA(Particles)
    auto histEtaB = ptEtaPhiMDF.Define("B_eta", "LHEPart_eta[2]").Histo1D({"histEtaB", "b;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "B_eta");
    auto histEtaBBar = ptEtaPhiMDF.Define("BBar_eta", "LHEPart_eta[5]").Histo1D({"histEtaBBar", "#bar{b};#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "BBar_eta");

    auto histEtaQ = ptEtaPhiMDF.Define("Q_eta", selectQ,{"isWPlusHadronic","LHEPart_eta"}).Histo1D({"histEtaQ", "q;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "Q_eta");
    auto histEtaQBar = ptEtaPhiMDF.Define("QBar_eta", selectQBar,{"isWPlusHadronic","LHEPart_eta"}).Histo1D({"histEtaQBar", "#bar{q};#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "QBar_eta");
    auto histEtaLept = ptEtaPhiMDF.Define("Lept_eta", selectLept,{"isWPlusHadronic","LHEPart_eta"}).Histo1D({"histEtaLept", "l;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "Lept_eta");

    StackPlotter etaParticles({histEtaB, histEtaBBar, histEtaQ, histEtaQBar, histEtaLept}, "#eta", "#eta", imageSaveFolder+"/eta/etaParticles.png");

    etaParticles.SetLegendPos({0.78, 0.6, 0.9, 0.9});
    etaParticles.SetPalette(55);
    etaParticles.SetDrawOpt("hist PMC PLC nostack");
    etaParticles.SetLineWidth(3);

#pragma endregion ETA(Particles)
#pragma endregion Particles

#pragma region Leading

#pragma region PT in PT(Leading)
    float ptMaxLeading = 350;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_pt", "leading(LHEPart_pt,isWPlusHadronic)");

    auto histLeadingFirstPt = ptEtaPhiMDF.Define("Leading_firstPt", "Leading_pt[0]").Histo1D({"histLeadingPt", "Leading p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_firstPt");

    auto histLeadingSecondPt = ptEtaPhiMDF.Define("Leading_secondPt", "Leading_pt[1]").Histo1D({"histLeadingSecondPt", "Second p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_secondPt");

    auto histLeadingThirdPt = ptEtaPhiMDF.Define("Leading_thirdPt", "Leading_pt[2]").Histo1D({"histLeadingThirdPt", "Third p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_thirdPt");

    auto histLeadingFourthPt = ptEtaPhiMDF.Define("Leading_fourthPt", "Leading_pt[3]").Histo1D({"histLeadingFourthPt", "Fourth p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_fourthPt");


    StackPlotter leadingPt({histLeadingFirstPt, histLeadingSecondPt, histLeadingThirdPt, histLeadingFourthPt}, "Leading p_{T}", "p_{T} [GeV]", imageSaveFolder+"/pt/leadingPt.png");

#pragma endregion PT in PT(Leading)
#pragma region PTpdgId in PT(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_ptPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_pt,isWPlusHadronic)");

    auto histLeadingFirstPtPdgId = ptEtaPhiMDF.Define("Leading_firstPtPdgId", "Leading_ptPdgId[0]").Histo1D({"histLeadingPtPdgId", "Leading;pdgId;Events",4,2,6}, "Leading_firstPtPdgId");

    auto histLeadingSecondPtPdgId = ptEtaPhiMDF.Define("Leading_secondPtPdgId", "Leading_ptPdgId[1]").Histo1D({"histLeadingSecondPtPdgId", "Second;pdgId;Events",4,2,6}, "Leading_secondPtPdgId");

    auto histLeadingThirdPtPdgId = ptEtaPhiMDF.Define("Leading_thirdPtPdgId", "Leading_ptPdgId[2]").Histo1D({"histLeadingThirdPtPdgId", "Third;pdgId;Events",4,2,6}, "Leading_thirdPtPdgId");

    auto histLeadingFourthPtPdgId = ptEtaPhiMDF.Define("Leading_fourthPtPdgId", "Leading_ptPdgId[3]").Histo1D({"histLeadingFourthPtPdgId", "Fourth;pdgId;Events",4,2,6}, "Leading_fourthPtPdgId");


    StackPlotter leadingPtPdgId({histLeadingFirstPtPdgId, histLeadingSecondPtPdgId, histLeadingThirdPtPdgId, histLeadingFourthPtPdgId}, "Ordered p_{T} for particle", "", imageSaveFolder+"/pt/leadingPtpdgId.png");

    leadingPtPdgId.Normalize();
    leadingPtPdgId.SetYLabel("Fraction");

    leadingPtPdgId.setPartLabel();
    leadingPtPdgId.SetDrawOpt("bar");
    leadingPtPdgId.SetStatsInLegend(false);

#pragma endregion PTpdgId(PT)(Leading)

#pragma region ETA in ETA(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_eta", "leading(LHEPart_eta,isWPlusHadronic,true)");

    auto histLeadingFirstEta = ptEtaPhiMDF.Define("Leading_firstEta", "Leading_eta[0]").Histo1D({"histLeadingEta", "Leading #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_firstEta");

    auto histLeadingSecondEta = ptEtaPhiMDF.Define("Leading_secondEta", "Leading_eta[1]").Histo1D({"histLeadingSecondEta", "Second #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_secondEta");

    auto histLeadingThirdEta = ptEtaPhiMDF.Define("Leading_thirdEta", "Leading_eta[2]").Histo1D({"histLeadingThirdEta", "Third #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_thirdEta");

    auto histLeadingFourthEta = ptEtaPhiMDF.Define("Leading_fourthEta", "Leading_eta[3]").Histo1D({"histLeadingFourthEta", "Fourth #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_fourthEta");


    StackPlotter leadingEta({histLeadingFirstEta, histLeadingSecondEta, histLeadingThirdEta, histLeadingFourthEta}, "Leading #eta", "#eta", imageSaveFolder+"/eta/leadingEta.png");

#pragma endregion ETA in ETA(Leading)
#pragma region ETApdgId in ETA(Leading)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_etaPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_eta,true)");

    auto histLeadingFirstEtaPdgId = ptEtaPhiMDF.Define("Leading_firstEtaPdgId", "Leading_etaPdgId[0]").Histo1D({"histLeadingEtaPdgId", "Leading;pdgId;Events",4,2,6}, "Leading_firstEtaPdgId");

    auto histLeadingSecondEtaPdgId = ptEtaPhiMDF.Define("Leading_secondEtaPdgId", "Leading_etaPdgId[1]").Histo1D({"histLeadingSecondEtaPdgId", "Second;pdgId;Events",4,2,6}, "Leading_secondEtaPdgId");

    auto histLeadingThirdEtaPdgId = ptEtaPhiMDF.Define("Leading_thirdEtaPdgId", "Leading_etaPdgId[2]").Histo1D({"histLeadingThirdEtaPdgId", "Third;pdgId;Events",4,2,6}, "Leading_thirdEtaPdgId");

    auto histLeadingFourthEtaPdgId = ptEtaPhiMDF.Define("Leading_fourthEtaPdgId", "Leading_etaPdgId[3]").Histo1D({"histLeadingFourthEtaPdgId", "Fourth;pdgId;Events",4,2,6}, "Leading_fourthEtaPdgId");


    StackPlotter leadingEtaPdgId({histLeadingFirstEtaPdgId, histLeadingSecondEtaPdgId, histLeadingThirdEtaPdgId, histLeadingFourthEtaPdgId}, "Ordered #eta for particles", "", imageSaveFolder+"/eta/leadingEtapdgId.png");

    leadingEtaPdgId.Normalize();
    leadingEtaPdgId.SetYLabel("Fraction");

    leadingEtaPdgId.setPartLabel();
    leadingEtaPdgId.SetDrawOpt("bar");
    leadingEtaPdgId.SetStatsInLegend(false);

#pragma endregion ETApdgId(Leading)

#pragma region ETA ordered in PT (Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("etaOrderedInPt", "orderAccordingToVec(LHEPart_eta,LHEPart_pt,isWPlusHadronic,true)");

    auto histLeadingFirstEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_firstEtaOrderedInPt", "etaOrderedInPt[0]").Histo1D({"histLeadingEtaOrderedInPt", "Leading p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_firstEtaOrderedInPt");

    auto histLeadingSecondEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_secondEtaOrderedInPt", "etaOrderedInPt[1]").Histo1D({"histLeadingSecondEtaOrderedInPt", "Second p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_secondEtaOrderedInPt");

    auto histLeadingThirdEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_thirdEtaOrderedInPt", "etaOrderedInPt[2]").Histo1D({"histLeadingThirdEtaOrderedInPt", "Third p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_thirdEtaOrderedInPt");

    auto histLeadingFourthEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_fourthEtaOrderedInPt", "etaOrderedInPt[3]").Histo1D({"histLeadingFourthEtaOrderedInPt", "Fourth p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_fourthEtaOrderedInPt");


    StackPlotter etaOrderedInPt({histLeadingFirstEtaOrderedInPt, histLeadingSecondEtaOrderedInPt, histLeadingThirdEtaOrderedInPt, histLeadingFourthEtaOrderedInPt}, "#eta ordered in p_{T}", "#eta", imageSaveFolder+"/eta/EtaOrderedInPt.png");

#pragma endregion ETA ordered in PT (Leading)

#pragma region PT ordered in ETA (Leading)
    //! NB PT ordered in the absolute value of eta
    ptEtaPhiMDF = ptEtaPhiMDF.Define("ptOrderedInEta", "orderAccordingToVec(LHEPart_pt,LHEPart_eta,isWPlusHadronic,true)");

    auto histLeadingFirstPtOrderedInEta = ptEtaPhiMDF.Define("Leading_firstPtOrderedInEta", "ptOrderedInEta[0]").Histo1D({"histLeadingPtOrderedInEta", "Leading #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_firstPtOrderedInEta");

    auto histLeadingSecondPtOrderedInEta = ptEtaPhiMDF.Define("Leading_secondPtOrderedInEta", "ptOrderedInEta[1]").Histo1D({"histLeadingSecondPtOrderedInEta", "Second #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_secondPtOrderedInEta");

    auto histLeadingThirdPtOrderedInEta = ptEtaPhiMDF.Define("Leading_thirdPtOrderedInEta", "ptOrderedInEta[2]").Histo1D({"histLeadingThirdPtOrderedInEta", "Third #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_thirdPtOrderedInEta");

    auto histLeadingFourthPtOrderedInEta = ptEtaPhiMDF.Define("Leading_fourthPtOrderedInEta", "ptOrderedInEta[3]").Histo1D({"histLeadingFourthPtOrderedInEta", "Fourth #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_fourthPtOrderedInEta");


    StackPlotter ptOrderedInEta({histLeadingFirstPtOrderedInEta, histLeadingSecondPtOrderedInEta, histLeadingThirdPtOrderedInEta, histLeadingFourthPtOrderedInEta}, "p_{T} ordered in |#eta|", "p_{T} [GeV]", imageSaveFolder+"/pt/PtOrderedInEta.png");

#pragma endregion PT ordered in ETA (Leading)

#pragma endregion Leading

#pragma region DELTA
    std::vector strPart{"B", "Q", "Qbar", "Bbar", "Lept"};
    std::vector<std::tuple<std::string, std::string>> strPartCoupleVec;

    for (auto strCouple : iter::product<2>(strPart)) {
        if (std::get<0>(strCouple) != std::get<1>(strCouple)) {
            strPartCoupleVec.push_back(strCouple);
        }
    }
    std::unordered_map<std::string, std::string> strPartLatex{
        {"B", "b"},
        {"Q", "q"},
        {"Qbar", "#bar{q}"},
        {"Bbar", "#bar{b}"},
        {"Lept", "l"},
    };
#pragma region ETA(DELTA)
    std::vector<RResultPtr<::TH1D>> histDeltaEtaVec;
    for(auto strPartCouple : strPartCoupleVec){
        std::string part1=std::get<0>(strPartCouple);
        std::string part2=std::get<1>(strPartCouple);
        std::string columnName = "DeltaEta";
        columnName += part1 + part2;

        std::string functionString = "deltaEta(LHEPart_eta,isWPlusHadronic,";
        functionString += "\"" + part1 + "\"" + "," + "\""+part2 + "\")";


        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName = "histDeltaEta";
        histName += part1 + part2;

        std::string titleXLabYLab = strPartLatex[part2];
        titleXLabYLab += ";#Delta#eta;Counts";

        histDeltaEtaVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsEta, EtaMin, EtaMax}, columnName.c_str()));
    }




    StackPlotter deltaEtaB({histDeltaEtaVec[0], histDeltaEtaVec[1], histDeltaEtaVec[2], histDeltaEtaVec[3]},"#Delta#eta b", "#Delta#eta", imageSaveFolder+"/eta/deltaEtaB.png");
    StackPlotter deltaEtaQ({histDeltaEtaVec[4], histDeltaEtaVec[5], histDeltaEtaVec[6], histDeltaEtaVec[7]}, "#Delta#eta q", "#Delta#eta", imageSaveFolder+"/eta/deltaEtaQ.png");
    StackPlotter deltaEtaQBar({histDeltaEtaVec[8], histDeltaEtaVec[9], histDeltaEtaVec[10], histDeltaEtaVec[11]}, "#Delta#eta #bar{q}", "#Delta#eta", imageSaveFolder+"/eta/deltaEtaQBar.png");
    StackPlotter deltaEtaBBar({histDeltaEtaVec[12], histDeltaEtaVec[13], histDeltaEtaVec[14], histDeltaEtaVec[15]}, "#Delta#eta #bar{b}", "#Delta#eta", imageSaveFolder+"/eta/deltaEtaBBar.png");
    StackPlotter deltaEtaLept({histDeltaEtaVec[16], histDeltaEtaVec[17], histDeltaEtaVec[18], histDeltaEtaVec[19]}, "#Delta#eta l", "#Delta#eta", imageSaveFolder+"/eta/deltaEtaLept.png");




#pragma endregion ETA(DELTA)


#pragma region PHI(DELTA)
    std::vector<RResultPtr<::TH1D>> histDeltaPhiVec;
    for (auto strPartCouple : strPartCoupleVec) {
        std::string part1 = std::get<0>(strPartCouple);
        std::string part2 = std::get<1>(strPartCouple);
        std::string columnName = "DeltaPhi";
        columnName += part1 + part2;

        std::string functionString = "deltaPhi(LHEPart_phi,isWPlusHadronic,";
        functionString += "\"" + part1 + "\"" + "," + "\"" + part2 + "\")";

        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName = "histDeltaPhi";
        histName += part1 + part2;

        std::string titleXLabYLab = strPartLatex[part2];
        titleXLabYLab += ";#Delta#phi;Counts";

        histDeltaPhiVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsPhi, phiMin, phiMax}, columnName.c_str()));
    }

    StackPlotter deltaPhiB({histDeltaPhiVec[0], histDeltaPhiVec[1], histDeltaPhiVec[2], histDeltaPhiVec[3]}, "#Delta#phi b", "#Delta#phi", imageSaveFolder+"/phi/deltaPhiB.png");
    StackPlotter deltaPhiQ({histDeltaPhiVec[4], histDeltaPhiVec[5], histDeltaPhiVec[6], histDeltaPhiVec[7]}, "#Delta#phi q", "#Delta#phi", imageSaveFolder+"/phi/deltaPhiQ.png");
    StackPlotter deltaPhiQBar({histDeltaPhiVec[8], histDeltaPhiVec[9], histDeltaPhiVec[10], histDeltaPhiVec[11]}, "#Delta#phi #bar{q}", "#Delta#phi", imageSaveFolder+"/phi/deltaPhiQBar.png");
    StackPlotter deltaPhiBBar({histDeltaPhiVec[12], histDeltaPhiVec[13], histDeltaPhiVec[14], histDeltaPhiVec[15]}, "#Delta#phi #bar{b}", "#Delta#phi", imageSaveFolder+"/phi/deltaPhiBBar.png");
    StackPlotter deltaPhiLept({histDeltaPhiVec[16], histDeltaPhiVec[17], histDeltaPhiVec[18], histDeltaPhiVec[19]}, "#Delta#phi l", "#Delta#phi", imageSaveFolder+"/phi/deltaPhiLept.png");


#pragma endregion PHI(DELTA)


#pragma region R(DELTA)
    std::vector<RResultPtr<::TH1D>> histDeltaRVec;

    for (auto strPartCouple : strPartCoupleVec) {
        std::string part1 = std::get<0>(strPartCouple);
        std::string part2 = std::get<1>(strPartCouple);
        std::string columnName = "DeltaR";
        columnName += part1 + part2;

        std::string funcArg1 = "DeltaPhi";
        funcArg1 += part1 + part2;

        std::string funcArg2 = "DeltaEta";
        funcArg2 += part1 + part2;

        std::string functionString = "deltaR(";
        functionString += funcArg1 + "," + funcArg2 + ")";
        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName = "histDeltaR";
        histName += part1 + part2;

        std::string titleXLabYLab = strPartLatex[part2];
        titleXLabYLab += ";#DeltaR;Counts";

        histDeltaRVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsEta, 0, EtaMax}, columnName.c_str()));
    }


    StackPlotter deltaRB({histDeltaRVec[0], histDeltaRVec[1], histDeltaRVec[2], histDeltaRVec[3]}, "#DeltaR b", "#DeltaR", imageSaveFolder+"/r/deltaRB.png");
    StackPlotter deltaRQ({histDeltaRVec[4], histDeltaRVec[5], histDeltaRVec[6], histDeltaRVec[7]}, "#DeltaR q", "#DeltaR", imageSaveFolder+"/r/deltaRQ.png");
    StackPlotter deltaRQBar({histDeltaRVec[8], histDeltaRVec[9], histDeltaRVec[10], histDeltaRVec[11]}, "#DeltaR #bar{q}", "#DeltaR", imageSaveFolder+"/r/deltaRQBar.png");
    StackPlotter deltaRBBar({histDeltaRVec[12], histDeltaRVec[13], histDeltaRVec[14], histDeltaRVec[15]}, "#DeltaR #bar{b}", "#DeltaR", imageSaveFolder+"/r/deltaRBBar.png");
    StackPlotter deltaRLept({histDeltaRVec[16], histDeltaRVec[17], histDeltaRVec[18], histDeltaRVec[19]}, "#DeltaR l", "#DeltaR", imageSaveFolder+"/r/deltaRLept.png");




#pragma endregion R(DELTA)

//! This is a real mess, rewrite it from scratch
#pragma region RMin(delta)
    std::vector<RResultPtr<::TH1D>> histDeltaRMinVec;
    for(auto &part1: strPart){

        //DeltaRMin{Part}
        std::string columnName = "DeltaRMin";
        columnName += part1;

        //DeltaRMin(DeltaR{Part1}{Part2}, ...)
        std::string functionString = "Min(";
        for(auto &part2: strPart){
            if(part1 != part2){
                functionString += "DeltaR";
                functionString += part1;
                functionString += part2;
                functionString += ",";
            }
        }
        functionString.pop_back();
        functionString += ")";

        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());


//-----------------------------------------------------
        std::string columnPartName = columnName + "Mask";

        std::string functionPartString="Mask";
        functionPartString+=functionString;
        functionPartString.pop_back();
        functionPartString+=",\"";
        functionPartString+=part1;
        functionPartString += "\")";



        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnPartName.c_str(), functionPartString.c_str());
//-----------------------------------------------------
        std::unordered_map<std::string, int> minMask = {
            {"B", 1},
            {"Q", 2},
            {"Qbar", 3},
            {"Bbar", 4},
            {"Lept", 5}
        };
        for (auto &part2 : strPart) {
            if (part1 != part2) {
                std::string histName = "histDeltaRMin";
                histName += part1;
                histName += part2;

                std::string titleXLabYLab = strPartLatex[part2];
                titleXLabYLab += ";#DeltaR_{min};Counts";

                std::string filterString=columnPartName+"=="+std::to_string(minMask[part2]);

                histDeltaRMinVec.push_back(ptEtaPhiMDF.Filter(filterString.c_str()).Histo1D({histName.c_str(), strPartLatex[part2].c_str(), nBinsEta, 0, EtaMax}, columnName.c_str()));
            }
        }
    }

    StackPlotter deltaRMinB({histDeltaRMinVec[0], histDeltaRMinVec[1], histDeltaRMinVec[2], histDeltaRMinVec[3]}, "#DeltaR_{min} b", "#DeltaR_{min}", imageSaveFolder+"/r/deltaRMinB.png");

    StackPlotter deltaRMinQ({histDeltaRMinVec[4], histDeltaRMinVec[5], histDeltaRMinVec[6], histDeltaRMinVec[7]}, "#DeltaR_{min} q", "#DeltaR_{min}", imageSaveFolder+"/r/deltaRMinQ.png");
    StackPlotter deltaRMinQBar({histDeltaRMinVec[8], histDeltaRMinVec[9], histDeltaRMinVec[10], histDeltaRMinVec[11]}, "#DeltaR_{min} #bar{q}", "#DeltaR_{min}", imageSaveFolder+"/r/deltaRMinQBar.png");
    StackPlotter deltaRMinBBar({histDeltaRMinVec[12], histDeltaRMinVec[13], histDeltaRMinVec[14], histDeltaRMinVec[15]}, "#DeltaR_{min} #bar{b}", "#DeltaR_{min}", imageSaveFolder+"/r/deltaRMinBBar.png");
    StackPlotter deltaRMinLept({histDeltaRMinVec[16], histDeltaRMinVec[17], histDeltaRMinVec[18], histDeltaRMinVec[19]}, "#DeltaR_{min} l", "#DeltaR_{min}", imageSaveFolder+"/r/deltaRMinLept.png");

    deltaRMinB.SetDrawOpt("hist");
    deltaRMinBBar.SetDrawOpt("hist");
    deltaRMinQ.SetDrawOpt("hist");
    deltaRMinQBar.SetDrawOpt("hist");
    deltaRMinLept.SetDrawOpt("hist");
#pragma endregion RMin (DELTA)


#pragma region absolute RMin(delta)

    auto histRMin=ptEtaPhiMDF.Define("RMin","std::min({DeltaRBQ,DeltaRBQbar,DeltaRBBbar,DeltaRBLept,DeltaRQQbar,DeltaRQBbar,DeltaRQLept,DeltaRQbarBbar,DeltaRQbarLept,DeltaRBbarLept})").Histo1D({"histR", "#DeltaR_{min}", nBinsEta, 0, EtaMax}, "RMin");

    auto histRMinPart = ptEtaPhiMDF.Define("RMinPart", "ARGMIN({DeltaRBQ,DeltaRBQbar,DeltaRBBbar,DeltaRBLept,DeltaRQQbar,DeltaRQBbar,DeltaRQLept,DeltaRQbarBbar,DeltaRQbarLept,DeltaRBbarLept})").Histo1D({"histRMinPart", "#DeltaR_{min}", 10, 0, 10}, "RMinPart");

    StackPlotter rMin({histRMin}, "#DeltaR_{min}", "#DeltaR_{min}", imageSaveFolder+"/r/rMin.png");
    StackPlotter rMinPart({histRMinPart}, "#DeltaR_{min}", "", imageSaveFolder+"/r/rMinPart.png");

    rMin.SetLegendPos({0.7, 0.7, 0.92, 0.81});
    rMin.DrawVerticalLine(0.4);
    rMin.Normalize();
    rMin.SetYLabel("Event Fraction");

    rMinPart.SetDrawOpt("hist");
    rMinPart.SetBinLabel(1, "bq");
    rMinPart.SetBinLabel(2, "b#bar{q}");
    rMinPart.SetBinLabel(3, "b#bar{b}");
    rMinPart.SetBinLabel(4, "bl");
    rMinPart.SetBinLabel(5, "q#bar{q}");
    rMinPart.SetBinLabel(6, "q#bar{b}");
    rMinPart.SetBinLabel(7, "ql");
    rMinPart.SetBinLabel(8, "#bar{q}#bar{b}");
    rMinPart.SetBinLabel(9, "#bar{q}l");
    rMinPart.SetBinLabel(10, "#bar{b}l");

    rMinPart.SetStatsInLegend(false);
    rMinPart.Normalize();
    rMinPart.SetYLabel("Fraction");

    rMinPart.SetLegendPos({0.7,0.7,0.81,0.81});

#pragma endregion absolute RMin (DELTA)

#pragma endregion DELTA

#pragma region PLOT

    std::vector<StackPlotter *> stackCollection{

        &ttbarMass,
        &ttbarMassWide,
        &ttbarEta,
        &ttbarPt,
        &tLeptHadMass,
        &tLeptHadMassWide,
        &tLeptHadEta,
        &tLeptHadPt,
        &WPMMass,
        &WPMMassWide,
        &WPMPt,
        &WPMEta,
        &WLeptHadMass,
        &WLeptHadMassWide,
        &WLeptHadPt,
        &WLeptHadEta,
        &WLeptHadPt,



//---------------------------

        &etaParticles,
        &ptParticles,

        &leadingPt,
        &leadingPtPdgId,

        &leadingEta,
        &leadingEtaPdgId,

        &etaOrderedInPt,
        &ptOrderedInEta,


        &deltaEtaB,
        &deltaEtaQ,
        &deltaEtaQBar,
        &deltaEtaBBar,
        &deltaEtaLept,


        &deltaPhiB,
        &deltaPhiQ,
        &deltaPhiQBar,
        &deltaPhiBBar,
        &deltaPhiLept,


        &deltaRB,
        &deltaRQ,
        &deltaRQBar,
        &deltaRBBar,
        &deltaRLept,



        &deltaRMinB,
        &deltaRMinQ,
        &deltaRMinQBar,
        &deltaRMinBBar,
        &deltaRMinLept,

        &rMin,
        &rMinPart,

        &leptons,
        &jetCouple,
        &jetCoupleNormalized
    };

    for (auto v : stackCollection) {
        v->Save();
    }
#pragma endregion PLOT
    }
