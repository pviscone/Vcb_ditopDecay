#include "global.cpp"

//! this section is not future proof, it work only with this dataset (in which b=2,q=3,qbar=4,bbar=5,l=6)
void singlePart() {

#pragma region Header
    gStyle->SetFillStyle(1001);

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();

    // Draw "Preliminary"
    writeExtraText = false;
#pragma endregion Header

#pragma region Particles

#pragma region PT(Particles)
    auto histPtB = ptEtaPhiMDF.Define("B_pt", "LHEPart_pt[2]").Histo1D({"histPtB", "b;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "B_pt");
    auto histPtBBar = ptEtaPhiMDF.Define("BBar_pt", "LHEPart_pt[5]").Histo1D({"histPtBBar", "#bar{b};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "BBar_pt");
    auto histPtQ = ptEtaPhiMDF.Define("Q_pt", "LHEPart_pt[3]").Histo1D({"histPtQ", "q;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Q_pt");
    auto histPtQBar = ptEtaPhiMDF.Define("QBar_pt", "LHEPart_pt[4]").Histo1D({"histPtQBar", "#bar{q};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "QBar_pt");
    auto histPtLept = ptEtaPhiMDF.Define("Lept_pt", "LHEPart_pt[6]").Histo1D({"histPtL", "l;p_{T} [GeV];Counts", nBinsPt, ptMin, ptMax}, "Lept_pt");

    StackPlotter ptParticles({histPtB, histPtBBar, histPtQ, histPtQBar, histPtLept}, "p_{T}", "p_{T} [GeV]", "./images/pt/ptParticles.png");

    ptParticles.SetLegendPos({0.7, 0.6, 0.9, 0.9});
    ptParticles.SetPalette(55);
    ptParticles.SetDrawOpt("hist PMC PLC nostack");
    ptParticles.SetLineWidth(3);

#pragma endregion PT(Particles)
#pragma region ETA(Particles)
    auto histEtaB = ptEtaPhiMDF.Define("B_eta", "LHEPart_eta[2]").Histo1D({"histEtaB", "b;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "B_eta");
    auto histEtaQ = ptEtaPhiMDF.Define("Q_eta", "LHEPart_eta[3]").Histo1D({"histEtaQ", "q;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "Q_eta");
    auto histEtaQBar = ptEtaPhiMDF.Define("QBar_eta", "LHEPart_eta[4]").Histo1D({"histEtaQBar", "#bar{q};#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "QBar_eta");
    auto histEtaBBar = ptEtaPhiMDF.Define("BBar_eta", "LHEPart_eta[5]").Histo1D({"histEtaBBar", "#bar{b};#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "BBar_eta");
    auto histEtaLept = ptEtaPhiMDF.Define("Lept_eta", "LHEPart_eta[6]").Histo1D({"histEtaLept", "l;#eta;Counts", nBinsEtaSingle, EtaMin, EtaMax}, "Lept_eta");

    StackPlotter etaParticles({histEtaB, histEtaBBar, histEtaQ, histEtaQBar, histEtaLept}, "#eta", "#eta", "./images/eta/etaParticles.png");

    etaParticles.SetLegendPos({0.78, 0.6, 0.9, 0.9});
    etaParticles.SetPalette(55);
    etaParticles.SetDrawOpt("hist PMC PLC nostack");
    etaParticles.SetLineWidth(3);

#pragma endregion ETA(Particles)
#pragma endregion Particles

#pragma region Leading

#pragma region PT(Leading)
    float ptMaxLeading = 350;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_pt", "leading(LHEPart_pt)");

    auto histLeadingFirstPt = ptEtaPhiMDF.Define("Leading_firstPt", "Leading_pt[0]").Histo1D({"histLeadingPt", "Leading p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_firstPt");

    auto histLeadingSecondPt = ptEtaPhiMDF.Define("Leading_secondPt", "Leading_pt[1]").Histo1D({"histLeadingSecondPt", "Second p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_secondPt");

    auto histLeadingThirdPt = ptEtaPhiMDF.Define("Leading_thirdPt", "Leading_pt[2]").Histo1D({"histLeadingThirdPt", "Third p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_thirdPt");

    auto histLeadingFourthPt = ptEtaPhiMDF.Define("Leading_fourthPt", "Leading_pt[3]").Histo1D({"histLeadingFourthPt", "Fourth p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_fourthPt");

    StackPlotter leadingPt({histLeadingFirstPt, histLeadingSecondPt, histLeadingThirdPt, histLeadingFourthPt}, "Leading p_{T}", "p_{T} [GeV]", "./images/pt/leadingPt.png");

#pragma endregion PT(Leading)
#pragma region PTpdgId(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_ptPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_pt)");

    auto histLeadingFirstPtPdgId = ptEtaPhiMDF.Define("Leading_firstPtPdgId", "Leading_ptPdgId[0]").Histo1D({"histLeadingPtPdgId", "Leading;pdgId;Events", 2, 1, 3}, "Leading_firstPtPdgId");

    auto histLeadingSecondPtPdgId = ptEtaPhiMDF.Define("Leading_secondPtPdgId", "Leading_ptPdgId[1]").Histo1D({"histLeadingSecondPtPdgId", "Second;pdgId;Events", 2, 1, 3}, "Leading_secondPtPdgId");

    auto histLeadingThirdPtPdgId = ptEtaPhiMDF.Define("Leading_thirdPtPdgId", "Leading_ptPdgId[2]").Histo1D({"histLeadingThirdPtPdgId", "Third;pdgId;Events", 2, 1, 3}, "Leading_thirdPtPdgId");

    auto histLeadingFourthPtPdgId = ptEtaPhiMDF.Define("Leading_fourthPtPdgId", "Leading_ptPdgId[3]").Histo1D({"histLeadingFourthPtPdgId", "Fourth;pdgId;Events", 2, 1, 3}, "Leading_fourthPtPdgId");

    StackPlotter leadingPtPdgId({histLeadingFirstPtPdgId, histLeadingSecondPtPdgId, histLeadingThirdPtPdgId, histLeadingFourthPtPdgId}, "Leading p_{T} pdgId", "pdgId", "./images/pt/leadingPtpdgId.png");

    leadingPtPdgId.setQuarkTypeLabel();
    leadingPtPdgId.SetDrawOpt("bar");
    leadingPtPdgId.SetStatsInLegend(false);

#pragma endregion PTpdgId(PT)(Leading)

#pragma region ETA(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_eta", "leading(LHEPart_eta,true)");

    auto histLeadingFirstEta = ptEtaPhiMDF.Define("Leading_firstEta", "Leading_eta[0]").Histo1D({"histLeadingEta", "Leading #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_firstEta");

    auto histLeadingSecondEta = ptEtaPhiMDF.Define("Leading_secondEta", "Leading_eta[1]").Histo1D({"histLeadingSecondEta", "Second #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_secondEta");

    auto histLeadingThirdEta = ptEtaPhiMDF.Define("Leading_thirdEta", "Leading_eta[2]").Histo1D({"histLeadingThirdEta", "Third #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_thirdEta");

    auto histLeadingFourthEta = ptEtaPhiMDF.Define("Leading_fourthEta", "Leading_eta[3]").Histo1D({"histLeadingFourthEta", "Fourth #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_fourthEta");

    StackPlotter leadingEta({histLeadingFirstEta, histLeadingSecondEta, histLeadingThirdEta, histLeadingFourthEta}, "Leading #eta", "#eta", "./images/eta/leadingEta.png");
#pragma endregion ETA(Leading)
#pragma region ETApdgId(Leading)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_etaPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_eta,true)");

    auto histLeadingFirstEtaPdgId = ptEtaPhiMDF.Define("Leading_firstEtaPdgId", "Leading_etaPdgId[0]").Histo1D({"histLeadingEtaPdgId", "Leading;pdgId;Events", 2, 1, 3}, "Leading_firstEtaPdgId");

    auto histLeadingSecondEtaPdgId = ptEtaPhiMDF.Define("Leading_secondEtaPdgId", "Leading_etaPdgId[1]").Histo1D({"histLeadingSecondEtaPdgId", "Second;pdgId;Events", 2, 1, 3}, "Leading_secondEtaPdgId");

    auto histLeadingThirdEtaPdgId = ptEtaPhiMDF.Define("Leading_thirdEtaPdgId", "Leading_etaPdgId[2]").Histo1D({"histLeadingThirdEtaPdgId", "Third;pdgId;Events", 2, 1, 3}, "Leading_thirdEtaPdgId");

    auto histLeadingFourthEtaPdgId = ptEtaPhiMDF.Define("Leading_fourthEtaPdgId", "Leading_etaPdgId[3]").Histo1D({"histLeadingFourthEtaPdgId", "Fourth;pdgId;Events", 2, 1, 3}, "Leading_fourthEtaPdgId");

    StackPlotter leadingEtaPdgId({histLeadingFirstEtaPdgId, histLeadingSecondEtaPdgId, histLeadingThirdEtaPdgId, histLeadingFourthEtaPdgId}, "Leading #eta pdgId", "pdgId", "./images/eta/leadingEtapdgId.png");
    leadingEtaPdgId.setQuarkTypeLabel();
    leadingEtaPdgId.SetDrawOpt("bar");
    leadingEtaPdgId.SetStatsInLegend(false);

#pragma endregion ETApdgId(Leading)
#pragma endregion Leading

#pragma region DELTA
#pragma region ETA(DELTA)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaEtaBQ", "LHEPart_eta[2]-LHEPart_eta[3]").Define("DeltaEtaBQBar", "LHEPart_eta[2]-LHEPart_eta[4]").Define("DeltaEtaBLept", "LHEPart_eta[2]-LHEPart_eta[6]").Define("DeltaEtaBBarQ", "LHEPart_eta[5]-LHEPart_eta[3]").Define("DeltaEtaBBarQBar", "LHEPart_eta[5]-LHEPart_eta[4]").Define("DeltaEtaBBarLept", "LHEPart_eta[5]-LHEPart_eta[6]");

    auto histDeltaEtaBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBU", "q;#Delta#eta;Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBQ");
    auto histDeltaEtaBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBDBar", "#bar{q};#Delta#eta b;Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBQBar");
    auto histDeltaEtaBLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBLept", "l;#Delta#eta b;Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBLept");

    auto histDeltaEtaBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaEtaBBarQ", "q;#Delta#eta #bar{b};Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBBarQ");
    auto histDeltaEtaBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaEtaBBarQBar", "#bar{q};#Delta#eta #bar{b};Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBBarQBar");
    auto histDeltaEtaBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaEtaBBarLept", "l;#Delta#eta #bar{b};Counts", nBinsEta, EtaMin, EtaMax}, "DeltaEtaBBarLept");

    StackPlotter deltaEtaB({histDeltaEtaBQUp, histDeltaEtaBQDown, histDeltaEtaBLept}, "#Delta#eta b", "#Delta#eta", "./images/eta/deltaEtaB.png");
    StackPlotter deltaEtaBBar({histDeltaEtaBBarQUp, histDeltaEtaBBarQDown, histDeltaEtaBBarLept}, "#Delta#eta #bar{b}", "#Delta#eta", "./images/eta/deltaEtaBBar.png");

#pragma endregion ETA(DELTA)

#pragma region PHI(DELTA)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaPhiBQ", "deltaPhi(LHEPart_phi[2],LHEPart_phi[3])").Define("DeltaPhiBQBar", "deltaPhi(LHEPart_phi[2],LHEPart_phi[4])").Define("DeltaPhiBLept", "deltaPhi(LHEPart_phi[2],LHEPart_phi[6])").Define("DeltaPhiBBarQ", "deltaPhi(LHEPart_phi[5],LHEPart_phi[3])").Define("DeltaPhiBBarQBar", "deltaPhi(LHEPart_phi[5],LHEPart_phi[4])").Define("DeltaPhiBBarLept", "deltaPhi(LHEPart_phi[5],LHEPart_phi[6])");

    auto histDeltaPhiBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBU", "q;#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBQ");

    auto histDeltaPhiBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBDBar", "#bar{q};#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBQBar");
    auto histDeltaPhiBLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBLept", "l;#Delta#phi b;Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBLept");

    auto histDeltaPhiBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaPhiBBar", "q;#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarQ");
    auto histDeltaPhiBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaPhiBBarDBar", "#bar{q};#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarQBar");
    auto histDeltaPhiBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaPhiBBarLept", "l;#Delta#phi #bar{b};Counts", nBinsPhi, phiMin, phiMax}, "DeltaPhiBBarLept");

    StackPlotter deltaPhiB({histDeltaPhiBQUp, histDeltaPhiBQDown, histDeltaPhiBLept}, "#Delta#phi b", "#Delta#phi", "./images/phi/deltaPhiB.png");

    deltaPhiB.SetLegendPos({0.2, 0.74, 0.33, 0.86});
    StackPlotter deltaPhiBBar({histDeltaPhiBBarQUp, histDeltaPhiBBarQDown, histDeltaPhiBBarLept}, "#Delta#phi #bar{b}", "#Delta#phi ", "./images/phi/deltaPhiBBar.png");

    deltaPhiBBar.SetLegendPos({0.2, 0.74, 0.33, 0.86});

#pragma endregion PHI(DELTA)

#pragma region R(DELTA)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("DeltaRBQ", "deltaR(DeltaPhiBQ,DeltaEtaBQ)").Define("DeltaRBQBar", "deltaR(DeltaPhiBQBar,DeltaEtaBQBar)").Define("DeltaRBLept", "deltaR(DeltaPhiBLept,DeltaEtaBLept)").Define("DeltaRBBarQ", "deltaR(DeltaPhiBBarQ,DeltaEtaBBarQ)").Define("DeltaRBBarQBar", "deltaR(DeltaPhiBBarQBar,DeltaEtaBBarQBar)").Define("DeltaRBBarLept", "deltaR(DeltaPhiBBarLept,DeltaEtaBBarLept)");

    auto histDeltaRBQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBU", "q;#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBQ");
    auto histDeltaRBQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBDBar", "#bar{q};#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBQBar");
    auto histDeltaRBLept = ptEtaPhiMDF.Histo1D({"histDeltaRBLept", "l;#DeltaR b;Counts", nBinsR, RMin, RMax}, "DeltaRBLept");

    auto histDeltaRBBarQUp = ptEtaPhiMDF.Filter("LHEPart_pdgId[3]==2 || LHEPart_pdgId[3]==4").Histo1D({"histDeltaRBBarU", "q;#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarQ");

    auto histDeltaRBBarQDown = ptEtaPhiMDF.Filter("LHEPart_pdgId[4]==-1 || LHEPart_pdgId[4]==-3 || LHEPart_pdgId[4]==-5").Histo1D({"histDeltaRBBarDBar", "#bar{q};#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarQBar");
    auto histDeltaRBBarLept = ptEtaPhiMDF.Histo1D({"histDeltaRBarBLept", "l;#DeltaR #bar{b};Counts", nBinsR, RMin, RMax}, "DeltaRBBarLept");

    StackPlotter deltaRB({histDeltaRBQUp, histDeltaRBQDown, histDeltaRBLept}, "#DeltaR b", "#DeltaR", "./images/r/deltaRB.png");

    StackPlotter deltaRBBar({histDeltaRBBarQUp, histDeltaRBBarQDown, histDeltaRBBarLept}, "#DeltaR #bar{b}", "#DeltaR", "./images/r/deltaRBBar.png");

#pragma endregion R(DELTA)
#pragma endregion DELTA

#pragma region PLOT

    std::vector<StackPlotter *> stackCollection{
        &etaParticles,
        &ptParticles,
        &deltaEtaB,
        &deltaEtaBBar,
        &deltaPhiB,
        &deltaPhiBBar,
        &deltaRB,
        &deltaRBBar,

        &leadingPt,
        &leadingPtPdgId,

        &leadingEta,
        &leadingEtaPdgId

    };

    for (auto v : stackCollection) {
        v->Save();
    }
#pragma endregion PLOT
}