#include <iostream>
#include <ROOT/RDataFrame.hxx>
using namespace ROOT;

#include "muonUtils.h"

#include "../../utils/CMSStyle/CMS_lumi.C"
#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"

#include "../../utils/DfUtils.h"
#include "../../utils/HistUtils.h"



void muonSelection(std::string filename, std::string text, std::string imageSaveFolder) {
   gStyle->SetFillStyle(1001);

  //Draw "Preliminary"
  writeExtraText = true;
  extraText = "Preliminary";
  datasetText = text;

  ROOT::EnableImplicitMT();


  gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
  setTDRStyle();
  gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");

 

  RDataFrame fileDF("Events", filename,
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
                     "Jet_hadronFlavour"});

  auto MuonsFromWDF=fileDF.Filter("(LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)");

  auto TriggeredMuonsDF=MuonsFromWDF.Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4");

  auto LooseMuonsDF = TriggeredMuonsDF.Filter("Muon_looseId[0] && Muon_pfIsoId[0]>1");
  auto MediumMuonsDF= TriggeredMuonsDF.Filter("Muon_mediumId[0] && Muon_pfIsoId[0]>2");
  auto TightMuonsDF = TriggeredMuonsDF.Filter("Muon_tightId[0] && Muon_pfIsoId[0]>3");

  int totalTriggeredMuons = TriggeredMuonsDF.Count().GetValue();

  int totalLooseMuons=LooseMuonsDF.Count().GetValue();
  int totalMediumMuons=MediumMuonsDF.Count().GetValue();
  int totalTightMuons=TightMuonsDF.Count().GetValue();

  //tight/medium/loose w.r.t both identification and isolation
  std::cout<<"Total triggered muons: "<<totalTriggeredMuons<<std::endl;
  std::cout<<"Total loose muons: "<<totalLooseMuons<< "  Fraction:" << (float) totalLooseMuons/totalTriggeredMuons << std::endl;
  std::cout << "Total medium muons: " << totalMediumMuons << "  Fraction:" << (float) totalMediumMuons / totalTriggeredMuons << std::endl;
  std::cout << "Total tight muons: " << totalTightMuons << "  Fraction:" << (float) totalTightMuons / totalTriggeredMuons << std::endl;

  //Using loose muons
  LooseMuonsDF = LooseMuonsDF.Define("flavourOfJetWithMuon", "Jet_hadronFlavour[Jet_muonIdx1==0 && Jet_hadronFlavour!=0][0]");

  int cJetsWithMuon=LooseMuonsDF.Filter("flavourOfJetWithMuon==4").Count().GetValue();
  int bJetsWithMuon=LooseMuonsDF.Filter("flavourOfJetWithMuon==5").Count().GetValue();

  std::cout<<"b Jet with leading muon inside: "<< bJetsWithMuon <<"  Fraction of events: "<<(float) bJetsWithMuon/totalLooseMuons<<std::endl;
  std::cout<<"c Jet with leading muon inside: "<< cJetsWithMuon <<"  Fraction of events: "<<(float) cJetsWithMuon/totalLooseMuons<<std::endl;
  std::cout<<"Total jets with leading muon inside: "<< bJetsWithMuon+cJetsWithMuon <<"  Fraction of events: "<<(float) (bJetsWithMuon+cJetsWithMuon)/totalLooseMuons<<std::endl;

  LooseMuonsDF = LooseMuonsDF.Define("SlimmedJet_pt","Jet_pt[Jet_jetId>0 && Jet_puId>0]");

  LooseMuonsDF = LooseMuonsDF.Define("LeadingJetsWithoutMuon_pt", "FourJetsWithoutMuon(SlimmedJet_pt,Jet_muonIdx1)");

  auto histLeadingJetsWithoutMuon_pt = LooseMuonsDF.Define("LeadingJetWithoutMuon_pt", "LeadingJetsWithoutMuon_pt[0]").Histo1D({"LeadingJetsWithoutMuon_pt", "Leading", 100, 0,300}, "LeadingJetWithoutMuon_pt");

  auto histSecondJetsWithoutMuon_pt = LooseMuonsDF.Define("SecondJetWithoutMuon_pt", "LeadingJetsWithoutMuon_pt[1]").Histo1D({"SecondJetsWithoutMuon_pt", "Second", 100, 0, 300}, "SecondJetWithoutMuon_pt");

  auto histThirdJetsWithoutMuon_pt = LooseMuonsDF.Define("ThirdJetWithoutMuon_pt", "LeadingJetsWithoutMuon_pt[2]").Histo1D({"ThirdJetsWithoutMuon_pt", "Third", 100, 0, 300}, "ThirdJetWithoutMuon_pt");

  auto histFourthJetsWithoutMuon_pt = LooseMuonsDF.Define("FourthJetWithoutMuon_pt", "LeadingJetsWithoutMuon_pt[3]").Histo1D({"FourthJetsWithoutMuon_pt", "Fourth", 100, 0, 300}, "FourthJetWithoutMuon_pt");

  int cutPtJet20 = LooseMuonsDF.Filter("LeadingJetsWithoutMuon_pt[0]>20").Count().GetValue();
  int cutPtJet30 = LooseMuonsDF.Filter("LeadingJetsWithoutMuon_pt[0]>30").Count().GetValue();

  std::cout<<"Total events with fourth jet pT>20: "<<cutPtJet20<<"  Fraction of events: "<<(float) cutPtJet20/totalLooseMuons<<std::endl;
  std::cout<<"Total events with fourth jet pT>30: "<<cutPtJet30<<"  Fraction of events: "<<(float) cutPtJet30/totalLooseMuons<<std::endl;

  StackPlotter orderedJetsWithoutMuon({histLeadingJetsWithoutMuon_pt,histSecondJetsWithoutMuon_pt,histThirdJetsWithoutMuon_pt,histFourthJetsWithoutMuon_pt},"OrdedJetsWithoutMuon p_{t}","p_{t} [GeV]",imageSaveFolder+"/OrdedJetsWithoutMuon_pt.png");

  std::vector<StackPlotter *> stackCollection {
    &orderedJetsWithoutMuon
  };

  for (auto v : stackCollection) {
      v->Save();
  }
}