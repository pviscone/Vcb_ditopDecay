#include "global.cpp"

void reco(){
    //----------HEADER STYLE----------
    gStyle->SetFillStyle(1001);

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();

    // Draw "Preliminary"
    writeExtraText = true;
    datasetText = "TTJets_SingleLeptFromTbar_TuneCP5_13TeV";

    //-------------------------------------------------------------------------------------------------------
    //                                          Create the histograms

    //-----------------------------Masses-----------------------------//
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

    auto histWPlusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWPlus>0").Histo1D({"histWPlusJetDecay", "W^{+} qq decay; ;Counts", 9, 1, 9}, "jetCoupleWPlus");
    /*     auto histWMinusJetDecay = ptEtaPhiMDF.Filter("jetCoupleWMinus>0").Histo1D({"histWMinusJetDecay", "W^{-} jet decay; ;Counts", 9, 1, 9}, "jetCoupleWMinus"); */

    StackPlotter jetCouple({histWPlusJetDecay}, "W hadronic Decays", "W qq Decay", "./images/WHadronicDecay.png", true, false, true);

    jetCouple.GetValue();
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

    StackPlotter ttbarMass({histMTBar, histMT}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/Mttbar.png", true, true, false);
    StackPlotter tLeptHadMass({histMTHad, histMTLept}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHad.png", true, true);
    StackPlotter WPMMass({histMWPlus, histMWMinus}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinus.png", true, true);
    StackPlotter WLeptHadMass({histMWHad, histMWLept}, "M_{W#rightarrow q#bar{q} }/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHad.png", true, true);

    StackPlotter ttbarMassWide({histMTWide, histMTBarWide}, "M_{t}/ M_{#bar{t}}", "M_{t}  [GeV]", "./images/mass/MttbarWide.png", true);
    StackPlotter tLeptHadMassWide({histMTHadWide, histMTLeptWide}, "M_{t#rightarrow q#bar{q}}/ M_{t#rightarrow l#nu}", "M_{t} [GeV]", "./images/mass/MtLeptHadWide.png", true);
    StackPlotter WPMMassWide({histMWPlusWide, histMWMinusWide}, "M_{W^{+}}/ M_{W^{-}}", "M_{W} [GeV]", "./images/mass/MWPlusMinusWide.png", true);
    StackPlotter WLeptHadMassWide({histMWHadWide, histMWLeptWide}, "M_{W#rightarrow q#bar{q}}/ M_{W#rightarrow l#nu}", "M_{W} [GeV]", "./images/mass/MWLeptHadWide.png", true);

    StackPlotter ttbarEta({histEtaT, histEtaTBar}, "#eta_{t}/#eta_{#bar{t}}", "#eta_{t}", "./images/eta/EtaTTbar.png", true);
    StackPlotter tLeptHadEta({histEtaTHad, histEtaTLept}, "#eta_{t#rightarrow q#bar{q}} / #eta_{t#rightarrow l#nu}", "#eta_{t}", "./images/eta/EtaTLeptHad.png", true);
    StackPlotter WPMEta({histEtaWPlus, histEtaWMinus}, "#eta_{W^{+}}/#eta_{W^{-}}", "#eta_{W}", "./images/eta/EtaWPlusMinux.png", true);
    StackPlotter WLeptHadEta({histEtaWHad, histEtaWLept}, "#eta_{W#rightarrow q#bar{q}}/#eta_{W#rightarrow l#nu}", "#eta_{W}", "./images/eta/EtaWLeptHad.png", true);

    StackPlotter ttbarPt({histPtT, histPtTBar}, "p_{T}(t)/p_{T}(#bar{t})", "p_{T} [GeV]", "./images/pt/PtTTBar.png", true);
    StackPlotter tLeptHadPt({histPtTHad, histPtTLept}, "p_{T}(t#rightarrow q#bar{q})/p_{T}(t#rightarrow l#nu)", "p_{T} [GeV]", "./images/pt/PtTLeptHad.png", true);
    StackPlotter WPMPt({histPtWPlus, histPtWMinus}, "p_{T}(W^{+})/p_{T}(W^{-})", "p_{T} [GeV]", "./images/pt/PtWPlusMinus.png", true);
    StackPlotter WLeptHadPt({histPtWHad, histPtWLept}, "p_{T}(W#rightarrow q#bar{q})/p_{T}(W#rightarrow l#nu)", "p_{T} [GeV]", "./images/pt/PtWLeptHad.png", true);

    ttbarMass.SetNStatBox(2);
    tLeptHadMass.SetNStatBox(2);
    WPMMass.SetNStatBox(2);
    WLeptHadMass.SetNStatBox(2);


    std::vector<StackPlotter *> stackCollection {
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

            &jetCouple,
        };

    for (auto v : stackCollection) {
        v->Save();
    }
    }
