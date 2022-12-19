#include "global.cpp"
#include "../../../utils/itertools/product.hpp"
#include "../../../utils/itertools/zip.hpp"

//! this section is not future proof, it work only with this dataset (in which b=2,q=3,qbar=4,bbar=5,l=6)
void singlePart() {

#pragma region Header
    gStyle->SetFillStyle(1001);

    // Draw "Preliminary"
    writeExtraText = true;
    extraText = "Preliminary";
    datasetText = "TTJets_SingleLeptFromTbar_TuneCP5_13TeV";

    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../../utils/CMSStyle/CMS_lumi.C");
    TH1::SetDefaultSumw2();


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

#pragma region PT in PT(Leading)
    float ptMaxLeading = 350;

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_pt", "leading(LHEPart_pt)");

    auto histLeadingFirstPt = ptEtaPhiMDF.Define("Leading_firstPt", "Leading_pt[0]").Histo1D({"histLeadingPt", "Leading p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_firstPt");

    auto histLeadingSecondPt = ptEtaPhiMDF.Define("Leading_secondPt", "Leading_pt[1]").Histo1D({"histLeadingSecondPt", "Second p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_secondPt");

    auto histLeadingThirdPt = ptEtaPhiMDF.Define("Leading_thirdPt", "Leading_pt[2]").Histo1D({"histLeadingThirdPt", "Third p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_thirdPt");

    auto histLeadingFourthPt = ptEtaPhiMDF.Define("Leading_fourthPt", "Leading_pt[3]").Histo1D({"histLeadingFourthPt", "Fourth p_{T};p_{T} [GeV];Counts", nBinsPt, ptMin, ptMaxLeading}, "Leading_fourthPt");


    StackPlotter leadingPt({histLeadingFirstPt, histLeadingSecondPt, histLeadingThirdPt, histLeadingFourthPt}, "Leading p_{T}", "p_{T} [GeV]", "./images/pt/leadingPt.png");

#pragma endregion PT in PT(Leading)
#pragma region PTpdgId in PT(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_ptPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_pt)");

    auto histLeadingFirstPtPdgId = ptEtaPhiMDF.Define("Leading_firstPtPdgId", "Leading_ptPdgId[0]").Histo1D({"histLeadingPtPdgId", "Leading;pdgId;Events",4,2,6}, "Leading_firstPtPdgId");

    auto histLeadingSecondPtPdgId = ptEtaPhiMDF.Define("Leading_secondPtPdgId", "Leading_ptPdgId[1]").Histo1D({"histLeadingSecondPtPdgId", "Second;pdgId;Events",4,2,6}, "Leading_secondPtPdgId");

    auto histLeadingThirdPtPdgId = ptEtaPhiMDF.Define("Leading_thirdPtPdgId", "Leading_ptPdgId[2]").Histo1D({"histLeadingThirdPtPdgId", "Third;pdgId;Events",4,2,6}, "Leading_thirdPtPdgId");

    auto histLeadingFourthPtPdgId = ptEtaPhiMDF.Define("Leading_fourthPtPdgId", "Leading_ptPdgId[3]").Histo1D({"histLeadingFourthPtPdgId", "Fourth;pdgId;Events",4,2,6}, "Leading_fourthPtPdgId");


    StackPlotter leadingPtPdgId({histLeadingFirstPtPdgId, histLeadingSecondPtPdgId, histLeadingThirdPtPdgId, histLeadingFourthPtPdgId}, "Ordered p_{T} for particle", "", "./images/pt/leadingPtpdgId.png");

    leadingPtPdgId.Normalize();
    leadingPtPdgId.SetYLabel("Fraction");

    leadingPtPdgId.setPartLabel();
    leadingPtPdgId.SetDrawOpt("bar");
    leadingPtPdgId.SetStatsInLegend(false);

#pragma endregion PTpdgId(PT)(Leading)

#pragma region ETA in ETA(Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_eta", "leading(LHEPart_eta,true)");

    auto histLeadingFirstEta = ptEtaPhiMDF.Define("Leading_firstEta", "Leading_eta[0]").Histo1D({"histLeadingEta", "Leading #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_firstEta");

    auto histLeadingSecondEta = ptEtaPhiMDF.Define("Leading_secondEta", "Leading_eta[1]").Histo1D({"histLeadingSecondEta", "Second #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_secondEta");

    auto histLeadingThirdEta = ptEtaPhiMDF.Define("Leading_thirdEta", "Leading_eta[2]").Histo1D({"histLeadingThirdEta", "Third #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_thirdEta");

    auto histLeadingFourthEta = ptEtaPhiMDF.Define("Leading_fourthEta", "Leading_eta[3]").Histo1D({"histLeadingFourthEta", "Fourth #eta;#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_fourthEta");


    StackPlotter leadingEta({histLeadingFirstEta, histLeadingSecondEta, histLeadingThirdEta, histLeadingFourthEta}, "Leading #eta", "#eta", "./images/eta/leadingEta.png");

#pragma endregion ETA in ETA(Leading)
#pragma region ETApdgId in ETA(Leading)

    ptEtaPhiMDF = ptEtaPhiMDF.Define("Leading_etaPdgId", "leadingIdx(LHEPart_pdgId,LHEPart_eta,true)");

    auto histLeadingFirstEtaPdgId = ptEtaPhiMDF.Define("Leading_firstEtaPdgId", "Leading_etaPdgId[0]").Histo1D({"histLeadingEtaPdgId", "Leading;pdgId;Events",4,2,6}, "Leading_firstEtaPdgId");

    auto histLeadingSecondEtaPdgId = ptEtaPhiMDF.Define("Leading_secondEtaPdgId", "Leading_etaPdgId[1]").Histo1D({"histLeadingSecondEtaPdgId", "Second;pdgId;Events",4,2,6}, "Leading_secondEtaPdgId");

    auto histLeadingThirdEtaPdgId = ptEtaPhiMDF.Define("Leading_thirdEtaPdgId", "Leading_etaPdgId[2]").Histo1D({"histLeadingThirdEtaPdgId", "Third;pdgId;Events",4,2,6}, "Leading_thirdEtaPdgId");

    auto histLeadingFourthEtaPdgId = ptEtaPhiMDF.Define("Leading_fourthEtaPdgId", "Leading_etaPdgId[3]").Histo1D({"histLeadingFourthEtaPdgId", "Fourth;pdgId;Events",4,2,6}, "Leading_fourthEtaPdgId");


    StackPlotter leadingEtaPdgId({histLeadingFirstEtaPdgId, histLeadingSecondEtaPdgId, histLeadingThirdEtaPdgId, histLeadingFourthEtaPdgId}, "Ordered #eta for particles", "", "./images/eta/leadingEtapdgId.png");

    leadingEtaPdgId.Normalize();
    leadingEtaPdgId.SetYLabel("Fraction");

    leadingEtaPdgId.setPartLabel();
    leadingEtaPdgId.SetDrawOpt("bar");
    leadingEtaPdgId.SetStatsInLegend(false);

#pragma endregion ETApdgId(Leading)

#pragma region ETA ordered in PT (Leading)
    ptEtaPhiMDF = ptEtaPhiMDF.Define("etaOrderedInPt", "orderAccordingToVec(LHEPart_eta,LHEPart_pt,true)");

    auto histLeadingFirstEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_firstEtaOrderedInPt", "etaOrderedInPt[0]").Histo1D({"histLeadingEtaOrderedInPt", "Leading p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_firstEtaOrderedInPt");

    auto histLeadingSecondEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_secondEtaOrderedInPt", "etaOrderedInPt[1]").Histo1D({"histLeadingSecondEtaOrderedInPt", "Second p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_secondEtaOrderedInPt");

    auto histLeadingThirdEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_thirdEtaOrderedInPt", "etaOrderedInPt[2]").Histo1D({"histLeadingThirdEtaOrderedInPt", "Third p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_thirdEtaOrderedInPt");

    auto histLeadingFourthEtaOrderedInPt = ptEtaPhiMDF.Define("Leading_fourthEtaOrderedInPt", "etaOrderedInPt[3]").Histo1D({"histLeadingFourthEtaOrderedInPt", "Fourth p_{T};#eta;Counts", nBinsEta, 0, EtaMax}, "Leading_fourthEtaOrderedInPt");


    StackPlotter etaOrderedInPt({histLeadingFirstEtaOrderedInPt, histLeadingSecondEtaOrderedInPt, histLeadingThirdEtaOrderedInPt, histLeadingFourthEtaOrderedInPt}, "#eta ordered in p_{T}", "#eta", "./images/eta/EtaOrderedInPt.png");

#pragma endregion ETA ordered in PT (Leading)

#pragma region PT ordered in ETA (Leading)
    //! NB PT ordered in the absolute value of eta
    ptEtaPhiMDF = ptEtaPhiMDF.Define("ptOrderedInEta", "orderAccordingToVec(LHEPart_pt,LHEPart_eta,true)");

    auto histLeadingFirstPtOrderedInEta = ptEtaPhiMDF.Define("Leading_firstPtOrderedInEta", "ptOrderedInEta[0]").Histo1D({"histLeadingPtOrderedInEta", "Leading #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_firstPtOrderedInEta");

    auto histLeadingSecondPtOrderedInEta = ptEtaPhiMDF.Define("Leading_secondPtOrderedInEta", "ptOrderedInEta[1]").Histo1D({"histLeadingSecondPtOrderedInEta", "Second #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_secondPtOrderedInEta");

    auto histLeadingThirdPtOrderedInEta = ptEtaPhiMDF.Define("Leading_thirdPtOrderedInEta", "ptOrderedInEta[2]").Histo1D({"histLeadingThirdPtOrderedInEta", "Third #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_thirdPtOrderedInEta");

    auto histLeadingFourthPtOrderedInEta = ptEtaPhiMDF.Define("Leading_fourthPtOrderedInEta", "ptOrderedInEta[3]").Histo1D({"histLeadingFourthPtOrderedInEta", "Fourth #eta;p_{T};Counts", nBinsPt, 0, ptMax}, "Leading_fourthPtOrderedInEta");


    StackPlotter ptOrderedInEta({histLeadingFirstPtOrderedInEta, histLeadingSecondPtOrderedInEta, histLeadingThirdPtOrderedInEta, histLeadingFourthPtOrderedInEta}, "p_{T} ordered in #eta", "p_{T} [GeV]", "./images/pt/PtOrderedInEta.png");

#pragma endregion PT ordered in ETA (Leading)

#pragma endregion Leading

#pragma region DELTA
    std::vector idxPart{2, 3, 4, 5, 6};
    std::vector strPart{"B", "Q", "QBar", "BBar", "Lept"};
    std::vector<std::tuple<int, int>> idxVec;
    std::vector<std::tuple<std::string, std::string>> strPartVec;
    for (auto idx : iter::product<2>(idxPart)) {
        if (std::get<0>(idx) != std::get<1>(idx)) {
            idxVec.push_back(idx);
        }
    }
    for (auto str : iter::product<2>(strPart)) {
        if (std::get<0>(str) != std::get<1>(str)) {
            strPartVec.push_back(str);
        }
    }
    std::unordered_map<std::string, std::string> strPartLatex{
        {"B", "b"},
        {"Q", "q"},
        {"QBar", "#bar{q}"},
        {"BBar", "#bar{b}"},
        {"Lept", "l"},
    };
#pragma region ETA(DELTA)

    std::vector<RResultPtr<::TH1D>> histDeltaEtaVec;
    for (auto [idxTuple, strTuple] : iter::zip(idxVec, strPartVec)) {
        std::string columnName = "DeltaEta";
        columnName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string functionString = "LHEPart_eta[";
        functionString += std::to_string(std::get<0>(idxTuple)) + "]-LHEPart_eta[" + std::to_string(std::get<1>(idxTuple)) + "]";
        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName="histDeltaEta";
        histName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string titleXLabYLab = strPartLatex[std::get<1>(strTuple)];
        titleXLabYLab += ";#Delta#eta;Counts";

        histDeltaEtaVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsEta, EtaMin, EtaMax}, columnName.c_str()));
    }

    StackPlotter deltaEtaB({histDeltaEtaVec[0], histDeltaEtaVec[1], histDeltaEtaVec[2], histDeltaEtaVec[3]},"#Delta#eta b", "#Delta#eta", "./images/eta/deltaEtaB.png");
    StackPlotter deltaEtaQ({histDeltaEtaVec[4], histDeltaEtaVec[5], histDeltaEtaVec[6], histDeltaEtaVec[7]}, "#Delta#eta q", "#Delta#eta", "./images/eta/deltaEtaQ.png");
    StackPlotter deltaEtaQBar({histDeltaEtaVec[8], histDeltaEtaVec[9], histDeltaEtaVec[10], histDeltaEtaVec[11]}, "#Delta#eta #bar{q}", "#Delta#eta", "./images/eta/deltaEtaQBar.png");
    StackPlotter deltaEtaBBar({histDeltaEtaVec[12], histDeltaEtaVec[13], histDeltaEtaVec[14], histDeltaEtaVec[15]}, "#Delta#eta #bar{b}", "#Delta#eta", "./images/eta/deltaEtaBBar.png");
    StackPlotter deltaEtaLept({histDeltaEtaVec[16], histDeltaEtaVec[17], histDeltaEtaVec[18], histDeltaEtaVec[19]}, "#Delta#eta l", "#Delta#eta", "./images/eta/deltaEtaLept.png");




#pragma endregion ETA(DELTA)


#pragma region PHI(DELTA)

    std::vector<RResultPtr<::TH1D>> histDeltaPhiVec;
    for (auto [idxTuple, strTuple] : iter::zip(idxVec, strPartVec)) {
        std::string columnName = "DeltaPhi";
        columnName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string functionString = "deltaPhi(LHEPart_phi[";
        functionString += std::to_string(std::get<0>(idxTuple)) + "],LHEPart_phi[" + std::to_string(std::get<1>(idxTuple)) + "])";
        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName = "histDeltaPhi";
        histName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string titleXLabYLab = strPartLatex[std::get<1>(strTuple)];
        titleXLabYLab += ";#Delta#phi;Counts";

        histDeltaPhiVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsPhi, phiMin, phiMax}, columnName.c_str()));
    }

    StackPlotter deltaPhiB({histDeltaPhiVec[0], histDeltaPhiVec[1], histDeltaPhiVec[2], histDeltaPhiVec[3]}, "#Delta#phi b", "#Delta#phi", "./images/phi/deltaPhiB.png");
    StackPlotter deltaPhiQ({histDeltaPhiVec[4], histDeltaPhiVec[5], histDeltaPhiVec[6], histDeltaPhiVec[7]}, "#Delta#phi q", "#Delta#phi", "./images/phi/deltaPhiQ.png");
    StackPlotter deltaPhiQBar({histDeltaPhiVec[8], histDeltaPhiVec[9], histDeltaPhiVec[10], histDeltaPhiVec[11]}, "#Delta#phi #bar{q}", "#Delta#phi", "./images/phi/deltaPhiQBar.png");
    StackPlotter deltaPhiBBar({histDeltaPhiVec[12], histDeltaPhiVec[13], histDeltaPhiVec[14], histDeltaPhiVec[15]}, "#Delta#phi #bar{b}", "#Delta#phi", "./images/phi/deltaPhiBBar.png");
    StackPlotter deltaPhiLept({histDeltaPhiVec[16], histDeltaPhiVec[17], histDeltaPhiVec[18], histDeltaPhiVec[19]}, "#Delta#phi l", "#Delta#phi", "./images/phi/deltaPhiLept.png");


#pragma endregion PHI(DELTA) 


#pragma region R(DELTA)
    std::vector<RResultPtr<::TH1D>> histDeltaRVec;
    for (auto [idxTuple, strTuple] : iter::zip(idxVec, strPartVec)) {
        std::string columnName = "DeltaR";
        columnName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string funcArg1= "DeltaPhi";
        funcArg1+= std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string funcArg2 = "DeltaEta";
        funcArg2 += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string functionString = "deltaR(";
        functionString += funcArg1 + "," + funcArg2 + ")";
        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string histName = "histDeltaR";
        histName += std::get<0>(strTuple) + std::get<1>(strTuple);

        std::string titleXLabYLab = strPartLatex[std::get<1>(strTuple)];
        titleXLabYLab += ";#Delta#R;Counts";

        histDeltaRVec.push_back(ptEtaPhiMDF.Histo1D({histName.c_str(), titleXLabYLab.c_str(), nBinsEta, 0, EtaMax}, columnName.c_str()));
    }

    StackPlotter deltaRB({histDeltaRVec[0], histDeltaRVec[1], histDeltaRVec[2], histDeltaRVec[3]}, "#DeltaR b", "#DeltaR", "./images/r/deltaRB.png");
    StackPlotter deltaRQ({histDeltaRVec[4], histDeltaRVec[5], histDeltaRVec[6], histDeltaRVec[7]}, "#DeltaR q", "#DeltaR", "./images/r/deltaRQ.png");
    StackPlotter deltaRQBar({histDeltaRVec[8], histDeltaRVec[9], histDeltaRVec[10], histDeltaRVec[11]}, "#DeltaR #bar{q}", "#DeltaR", "./images/r/deltaRQBar.png");
    StackPlotter deltaRBBar({histDeltaRVec[12], histDeltaRVec[13], histDeltaRVec[14], histDeltaRVec[15]}, "#DeltaR #bar{b}", "#DeltaR", "./images/r/deltaRBBar.png");
    StackPlotter deltaRLept({histDeltaRVec[16], histDeltaRVec[17], histDeltaRVec[18], histDeltaRVec[19]}, "#DeltaR l", "#DeltaR", "./images/r/deltaRLept.png");




#pragma endregion R(DELTA)


#pragma region RMin(delta)
//NON servono più le tuple, e non serve idxVec, si fa tutto con colonne già definite
    std::vector<RResultPtr<::TH1D>> histDeltaRMinVec;
    for(auto &str1: strPart){
        std::string columnName = "DeltaRMin";
        columnName += str1;

        std::string functionString = "DeltaRMin(";

        for(auto &str2: strPart){
            if(str1 != str2){
                functionString += "DeltaR";
                functionString += str1;
                functionString += str2;
                functionString += ",";
            }
        }
        functionString.pop_back();
        functionString += ")";

        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnName.c_str(), functionString.c_str());

        std::string functionPartString="Part";
        functionPartString+=functionString;
        functionPartString.pop_back();
        functionPartString+=",";
        functionPartString+=strToPosString[str1];
        functionPartString+=")";

        std::string columnPartName="Part";
        columnPartName+=columnName;

        ptEtaPhiMDF = ptEtaPhiMDF.Define(columnPartName.c_str(), functionPartString.c_str());

        for (auto &str2 : strPart) {
            if (str1 != str2) {
                std::string histName = "histDeltaRMin";
                histName += str1;
                histName += str2;

                std::string titleXLabYLab = strPartLatex[str2];
                titleXLabYLab += ";#DeltaR_{min};Counts";

                std::string filterString=columnPartName+"=="+strToPosString[str2];

                histDeltaRMinVec.push_back(ptEtaPhiMDF.Filter(filterString.c_str()).Histo1D({histName.c_str(), strPartLatex[str2].c_str(), nBinsEta, 0, EtaMax}, columnName.c_str()));
            }
        }
    }

    StackPlotter deltaRMinB({histDeltaRMinVec[0], histDeltaRMinVec[1], histDeltaRMinVec[2], histDeltaRMinVec[3]}, "#DeltaR_{min} b", "#DeltaR_{min}", "./images/r/deltaRMinB.png");

    StackPlotter deltaRMinQ({histDeltaRMinVec[4], histDeltaRMinVec[5], histDeltaRMinVec[6], histDeltaRMinVec[7]}, "#DeltaR_{min} q", "#DeltaR_{min}", "./images/r/deltaRMinQ.png");
    StackPlotter deltaRMinQBar({histDeltaRMinVec[8], histDeltaRMinVec[9], histDeltaRMinVec[10], histDeltaRMinVec[11]}, "#DeltaR_{min} #bar{q}", "#DeltaR_{min}", "./images/r/deltaRMinQBar.png");
    StackPlotter deltaRMinBBar({histDeltaRMinVec[12], histDeltaRMinVec[13], histDeltaRMinVec[14], histDeltaRMinVec[15]}, "#DeltaR_{min} #bar{b}", "#DeltaR_{min}", "./images/r/deltaRMinBBar.png");
    StackPlotter deltaRMinLept({histDeltaRMinVec[16], histDeltaRMinVec[17], histDeltaRMinVec[18], histDeltaRMinVec[19]}, "#DeltaR_{min} l", "#DeltaR_{min}", "./images/r/deltaRMinLept.png");

    deltaRMinB.SetDrawOpt("hist");
    deltaRMinBBar.SetDrawOpt("hist");
    deltaRMinQ.SetDrawOpt("hist");
    deltaRMinQBar.SetDrawOpt("hist");
    deltaRMinLept.SetDrawOpt("hist");

#pragma endregion RMin (DELTA)



#pragma region absolute RMin(delta)

    auto histRMin=ptEtaPhiMDF.Define("RMin","std::min({DeltaRBQ,DeltaRBQBar,DeltaRBBBar,DeltaRBLept,DeltaRQQBar,DeltaRQBBar,DeltaRQLept,DeltaRQBarBBar,DeltaRQBarLept,DeltaRBBarLept})").Histo1D({"histR", "#DeltaR_{min}", nBinsEta, 0, EtaMax}, "RMin");

    auto histRMinPart = ptEtaPhiMDF.Define("RMinPart", "ARGMIN({DeltaRBQ,DeltaRBQBar,DeltaRBBBar,DeltaRBLept,DeltaRQQBar,DeltaRQBBar,DeltaRQLept,DeltaRQBarBBar,DeltaRQBarLept,DeltaRBBarLept})").Histo1D({"histRMinPart", "#DeltaR_{min}", 10, 0, 10}, "RMinPart");

    StackPlotter rMin({histRMin}, "#DeltaR_{min}", "#DeltaR_{min}", "./images/r/rMin.png");
    StackPlotter rMinPart({histRMinPart}, "#DeltaR_{min}", "", "./images/r/rMinPart.png");

    rMin.SetLegendPos({0.7, 0.7, 0.92, 0.81});
    rMin.DrawVerticalLine(0.4);

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
       &etaParticles,
        &ptParticles,

        &leadingPt,
        &leadingPtPdgId,

        &leadingEta,
        &leadingEtaPdgId,

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

        &etaOrderedInPt,
        &ptOrderedInEta,


        &deltaRMinB,
        &deltaRMinQ,
        &deltaRMinQBar,
        &deltaRMinBBar,
        &deltaRMinLept,
        &rMin,
        &rMinPart
    };

    for (auto v : stackCollection) {
        v->Save();
    }
#pragma endregion PLOT
        }
