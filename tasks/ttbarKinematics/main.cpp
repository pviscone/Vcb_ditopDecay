#include <iostream>
#include <unordered_map>

#include <TDatabasePDG.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TCanvas.h>

#include "../../utils/CMSStyle/CMS_lumi.h"
#include "../../utils/CMSStyle/tdrstyle.C"




//----------------------------------------------------------------------
//----------------------------------------------------------------------
//                         PDG DATABASE

//global pdg database
TDatabasePDG* pdgDatabase=new TDatabasePDG();


/**
 * @brief Function that, given the name of the particle, returns the PDG code
 * If the particle is not found, the function returns 0 (the Rootino)
 *
 * @param name particle name
 * @return int pdgId
 */
int pdg(const char * name){
    int id;
    if (id==0){
        std::cerr << "WARNING: there is a ROOTINO" << std::endl;
    }
    try{
        id=pdgDatabase->GetParticle(name)->PdgCode();
    } catch (std::exception &err){
        std::cerr << "The particle name: " << name << " does not exist" << std::endl;
        id=0;

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
std::string pdg(int id){
    std::string name;
    try{
        name=pdgDatabase->GetParticle(id)->GetName();
    } catch (std::exception &err){
        std::cerr << "The pdgId: "<< id << " does not exist" <<std::endl;
        name="Rootino";
    }
    return name;
}


/**
 * @brief Function that return the TParticlePDG object given the name of the particle
 * 
 * @param name particle name
 * @return TParticlePDG* 
 */
TParticlePDG* particle(const char * name){
    return pdgDatabase->GetParticle(name);
}


/**
 * @brief Function that return the TParticlePDG object given the pdgId of the particle
 *
 * @param name particle name
 * @return TParticlePDG*
 */
TParticlePDG* particle(int id){
    return pdgDatabase->GetParticle(id);
}


bool isQuark(int id){
    return (abs(id)<=8);
}

TLorentzVector getTLorentzVector(TTree* tree,int instance){
    TLorentzVector v;
    double pt=tree->GetLeaf("LHEPart_pt")->GetValue(instance);
    double eta=tree->GetLeaf("LHEPart_eta")->GetValue(instance);
    double phi=tree->GetLeaf("LHEPart_phi")->GetValue(instance);
    double mass=tree->GetLeaf("LHEPart_mass")->GetValue(instance);
    v.SetPtEtaPhiM(pt,eta,phi,mass);
    return v;
}


//----------------------------------------------------------------------
//----------------------------------------------------------------------



//----------------------------------------------------------------------
//----------------------------------------------------------------------
//                         MAIN LOOP OVER EVENTS

void macro(){
    ROOT::EnableImplicitMT();
    gROOT->LoadMacro("../../utils/CMSStyle/tdrstyle.C");
    setTDRStyle();
    gROOT->LoadMacro("../../utils/CMSStyle/CMS_lumi.C");


    TH1F* histMTLept=new TH1F("histMTLept","M_{t} leptonic;M_{t}\\  [GeV];Counts",60,140,200);
    TH1F* histMTHad=new TH1F("histMThad","M_{t} hadronic;M_{t}\\  [GeV];Counts",60,140,200);

    TH1F* histMTBarLept=new TH1F("histMTBarLept","M_{\\bar{t}}\\ leptonic;M_{\\bar{t}} \\ [GeV];Counts",100,100,250);
    TH1F* histMTBarHad=new TH1F("histMTBarhad","M_{\\bar{t}}\\ hadronic; M_{\\bar{t}} \\ [GeV];Counts",100,100,250);

    TH1F* histMWPlusLept=new TH1F("histMWPlusLept","M_{W^{+}} leptonic;M_{W^{+}} [GeV];Counts",100,0,200);
    TH1F* histMWPlusHad=new TH1F("histMWPlusHad","M_{W^{+}} hadronic;M_{W^{+}} [GeV];Counts",100,0,200);

    TH1F* histMWMinusLept=new TH1F("histMWMinusLept","M_{W^{-}} leptonic;M_{W^{-}} [GeV];Counts",100,0,200);
    TH1F* histMWMinusHad=new TH1F("histMWMinusHad","M_{W^{-}} hadronic;M_{W^{-}} [GeV];Counts",100,0,200);


    TH1F* histTPtLept=new TH1F("histTPtLept","t p_{t} leptonic;p_{t} [GeV];Counts",100,0,200);
    TH1F* histTPtHad=new TH1F("histTPtHad","t p_{t} hadronic;p_{t} [GeV];Counts",100,0,200);

    TH1F* histTBarPtLept=new TH1F("histTBarPtLept","\\bar{t} p_{t} leptonic; p_{t} [GeV];Counts",100,0,200);
    TH1F* histTBarPtHad=new TH1F("histTBarPtHad","\\bar{t} p_{t} hadronic;p_{\\bar{t}} [GeV];Counts",100,0,200);

    TH1F* histWPlusPtLept=new TH1F("histWPlusPtLept","W^{+} p_{t} leptonic;p_{t} [GeV];Counts",100,0,200);
    TH1F* histWPlusPtHad=new TH1F("histWPlusPtHad","W^{+} p_{t} hadronic;p_{t} [GeV];Counts",100,0,200);

    TH1F* histWMinusPtLept=new TH1F("histWMinusPtLept","W^{-} p_{t} leptonic;p_{t} [GeV];Counts",100,0,200);
    TH1F* histWMinusPtHad=new TH1F("histWMinusPtHad","W^{-} p_{t} hadronic;p_{t} [GeV];Counts",100,0,200);


    TH1F* histTEtaLept=new TH1F("histTEtaLept","t \\eta leptonic;\\eta;Counts",100,-10,10);
    TH1F* histTEtaHad=new TH1F("histTEtaHad","t \\eta hadronic;\\eta;Counts",100,-10,10);

    TH1F* histTBarEtaLept=new TH1F("histTBarEtaLept","\\bar{t} \\eta leptonic;\\eta;Counts",100,-10,10);
    TH1F* histTBarEtaHad=new TH1F("histTBarEtaHad","\\bar{t} \\eta hadronic;\\eta;Counts",100,-10,10);

    TH1F* histWPlusEtaLept=new TH1F("histWPlusEtaLept","W^{+} \\eta leptonic;\\eta;Counts",100,-10,10);
    TH1F* histWPlusEtaHad=new TH1F("histWPlusEtaHad","W^{+} \\eta hadronic;\\eta;Counts",100,-10,10);

    TH1F* histWMinusEtaLept=new TH1F("histWMinusEtaLept","W^{-} \\eta leptonic;\\eta;Counts",100,-10,10);
    TH1F* histWMinusEtaHad=new TH1F("histWMinusEtaHad","W^{-} \\eta hadronic;\\eta;Counts",100,-10,10);


    TH1I* histWPlusJetDecay=new TH1I("histWPlusJetDecay","W^{+} jet decay; ;Counts",10,0,10);
    TH1I* histWMinusJetDecay=new TH1I("histWMinusJetDecay","W^{-} jet decay; ;Counts",10,0,10);


    std::unordered_map<std::string, int> jetCoupleDictionary;
    jetCoupleDictionary["ud"]=1;
    jetCoupleDictionary["us"]=2;
    jetCoupleDictionary["ub"]=3;
    jetCoupleDictionary["db"]=4;
    jetCoupleDictionary["sb"]=5;
    jetCoupleDictionary["cb"]=6;
    jetCoupleDictionary["td"]=7;
    jetCoupleDictionary["ts"]=8;
    jetCoupleDictionary["tb"]=9;

    for(auto& couple : jetCoupleDictionary){
        histWPlusJetDecay->GetXaxis()->SetBinLabel(couple.second, couple.first.c_str());
        histWMinusJetDecay->GetXaxis()->SetBinLabel(couple.second, couple.first.c_str());
    }

    jetCoupleDictionary["du"]=1;
    jetCoupleDictionary["su"]=2;
    jetCoupleDictionary["bu"]=3;
    jetCoupleDictionary["bd"]=4;
    jetCoupleDictionary["sb"]=5;
    jetCoupleDictionary["bc"]=6;
    jetCoupleDictionary["dt"]=7;
    jetCoupleDictionary["st"]=8;
    jetCoupleDictionary["bt"]=9;

    std::vector<int> indexFromWPlus (2);
    std::vector<int> indexFromWMinus (2);
    int indexQFromT=3;
    int indexQBarFromTBar=4;

    TLorentzVector lorentzVectorWPlusLept;
    TLorentzVector lorentzVectorWPlusHad;
    TLorentzVector lorentzVectorWMinusLept;
    TLorentzVector lorentzVectorWMinusHad;
    TLorentzVector lorentzVectorTLept;
    TLorentzVector lorentzVectorTHad;
    TLorentzVector lorentzVectorTBarLept;
    TLorentzVector lorentzVectorTBarHad;
//setptetaphim

    double pt;
    double eta;
    double phi;
    double mass;


    TFile* file=new TFile("ttbar.root");
    TTree* tree=(TTree*)file->Get("Events");

    tree->SetBranchStatus("*",0);
    tree->SetBranchStatus("LHEPart*",1);

    int NofEvents=tree->GetEntries();
    NofEvents=100000;
    for(int eventNumber=0;eventNumber<NofEvents;eventNumber++){
        tree->GetEntry(eventNumber);

        int pdgId5=(tree->GetLeaf("LHEPart_pdgId"))->GetValue(5);
        if(particle(pdgId5)->Charge()>0.){
            indexFromWPlus[0]=5;
            indexFromWPlus[1]=6;
            indexFromWMinus[0]=7;
            indexFromWMinus[1]=8;
        } else {
            indexFromWPlus[0]=7;
            indexFromWPlus[1]=8;
            indexFromWMinus[0]=5;
            indexFromWMinus[1]=6;
        }
        if(isQuark(tree->GetLeaf("LHEPart_pdgId")->GetValue(indexFromWPlus[0]))){
            lorentzVectorWPlusHad=getTLorentzVector(tree, indexFromWPlus[0])+getTLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWMinusLept=getTLorentzVector(tree, indexFromWMinus[0])+getTLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTBarLept=getTLorentzVector(tree, indexQBarFromTBar)+lorentzVectorWMinusLept;
            lorentzVectorTHad=getTLorentzVector(tree, indexQFromT)+lorentzVectorWPlusHad;

            histMTBarLept->Fill(lorentzVectorTBarLept.M());
            histMTHad->Fill(lorentzVectorTHad.M());
            histMWPlusHad->Fill(lorentzVectorWPlusHad.M());
            histMWMinusLept->Fill(lorentzVectorWMinusLept.M());

        }else{
            lorentzVectorWPlusLept=getTLorentzVector(tree, indexFromWPlus[0])+getTLorentzVector(tree, indexFromWPlus[1]);
            lorentzVectorWMinusHad=getTLorentzVector(tree, indexFromWMinus[0])+getTLorentzVector(tree, indexFromWMinus[1]);

            lorentzVectorTBarHad=getTLorentzVector(tree, indexQBarFromTBar)+lorentzVectorWMinusHad;
            lorentzVectorTLept=getTLorentzVector(tree, indexQFromT)+lorentzVectorWPlusLept;
            histMTBarHad->Fill(lorentzVectorTBarHad.M());
            histMTLept->Fill(lorentzVectorTLept.M());
            histMWPlusLept->Fill(lorentzVectorWPlusLept.M());
            histMWMinusHad->Fill(lorentzVectorWMinusHad.M());
        }
    }


    TCanvas* c1=new TCanvas("c1","c1",800,600);
    TCanvas* c2=new TCanvas("c2","c2",800,600);
    TCanvas* c3=new TCanvas("c3","c3",800,600);
    TCanvas* c4=new TCanvas("c4","c4",800,600);
    TCanvas* c5=new TCanvas("c5","c5",800,600);
    TCanvas* c6=new TCanvas("c6","c6",800,600);
    TCanvas* c7=new TCanvas("c7","c7",800,600);
    TCanvas* c8=new TCanvas("c8","c8",800,600);
    gStyle->SetPalette(70);

    c1->cd();
    histMTHad->Draw("SAME HIST PLC PMC");
    histMTLept->Draw("SAME HIST PLC PMC");


    c1->BuildLegend();
    c1->SaveAs("./images/mass/histMTLept.png");


    c2->cd();
    histMTBarLept->Draw("SAME HIST PLC PMC");
    histMTBarHad->Draw("SAME HIST PLC PMC");



    c2->BuildLegend();
    c2->SaveAs("./images/mass/histMTBarLept.png");




}
