#include <ROOT/RDataFrame.hxx>
#include <iostream>
#include <regex>
using namespace ROOT;
using namespace ROOT::VecOps;
#include <string>
#include <iostream>
#include <filesystem>




float Met_eta(const RVec<float> &Lept_pt, const RVec<float> &Lept_eta, const RVec<float> &Lept_phi,const float &Met_pt, const float &Met_phi){
    float Muon_pt=Lept_pt[0];
    float Muon_phi=Lept_phi[0];
    float Muon_eta=Lept_eta[0];
    float Mw=80.385;
    float El2=pow(Muon_pt,2)*pow(cosh(Muon_eta),2);
    float pt_scalar_product=Met_pt*Muon_pt*cos(Met_phi-Muon_phi);
    float a = pow(Muon_pt,2);
    float b = -Muon_pt*sinh(Muon_eta)*(pow(Mw,2)+2*pt_scalar_product);
    float c = (-pow((pow(Mw,2)+2*pt_scalar_product),2)+4*El2*pow(Met_pt,2))/4;
    float delta= pow(b,2)-4*a*c;
    if(delta<0){
        delta=0;
    }
    float nu_eta=(-b+sqrt(delta))/(2*a);
    nu_eta=asinh(nu_eta/Met_pt);
    return nu_eta;
}

RVec<ROOT::Math::PtEtaPhiMVector> Jet_4Vector(
                                 const RVec<float> &Jet_pt,
                                 const RVec<float> &Jet_eta,
                                 const RVec<float> &Jet_phi,
                                 const RVec<float> &Jet_mass){

    RVec<ROOT::Math::PtEtaPhiMVector> Jet4V(Jet_pt.size());
    for(int i=0; i<Jet_pt.size();i++){
       Jet4V[i]=ROOT::Math::PtEtaPhiMVector(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i]);
    }
    return Jet4V;

}



RVec<float> TMass(const ROOT::Math::PtEtaPhiMVector &Mu4V,
                  const ROOT::Math::PtEtaPhiMVector &Nu4V,
                  const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V){
    RVec<float> tmass(Jet4V.size());
    for(int i=0; i<Jet4V.size();i++){
        tmass[i]=((Jet4V[i]+Mu4V+Nu4V).M());
    }
    return tmass;
}

RVec<float> WHad_mass(const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V){
    float Mw=80.385;
    int n=Jet4V.size();
    RVec<float> best_Wmass(n);
    for(int i=0;i<n;i++){
        RVec<float> temp_Wmass(n,0.);
        for(int j=0;j<n;j++){
            temp_Wmass[j]=(Jet4V[i]+Jet4V[j]).M();
        }
        int argBest_Wmass=ArgMin(abs(temp_Wmass-Mw));
        best_Wmass[i]=temp_Wmass[argBest_Wmass];
    }
    return best_Wmass;
}



RVec<float> THad_mass(const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V){
    float MT=173.1;
    int n=Jet4V.size();
    RVec<float> best_Tmass(n);
    for(int i=0;i<n;i++){
        RVec<float> temp_Tmass(n*n,0.);
        for(int j=0;j<n;j++){
            for(int k=0;k<n;k++){
                temp_Tmass[j*n+k]=(Jet4V[i]+Jet4V[j]+Jet4V[k]).M();      
            }
        }
        int argBest_Tmass=ArgMin(abs(temp_Tmass-MT));
        best_Tmass[i]=temp_Tmass[argBest_Tmass];
    }
    return best_Tmass;
}

//"./powheg/root_files"
void MuonCuts(std::string input,std::string output){
    std::vector<std::string> files_path;
    if(input.find(".root")==string::npos){
        for (const auto & entry : std::filesystem::directory_iterator()){
            files_path.push_back(entry.path());
        }
    }else{
        files_path.push_back(input);
    }
    EnableImplicitMT();
    
    RDataFrame df("Events",files_path,{"LHEPart_pdgId",
                                       "LHEPart_pt",
                                       "LHEPart_eta",
                                       "LHEPart_phi"
                                       "Muon_pt",
                                       "Muon_eta",
                                       "Muon_phi",
                                       "Muon_looseId",
                                       "Muon_pfIsoId",
                                       "Muon_charge",
                                       "nMuon",
                                       "MET_pt",
                                       "MET_phi",
                                       "Jet_pt",
                                       "Jet_eta",
                                       "Jet_phi",
                                       "Jet_mass",
                                       "Jet_muonIdx1",
                                       "Jet_jetId",
                                       "Jet_puId",
                                       "nJet",
                                       "Jet_btagDeepFlavB",
                                       "Jet_btagDeepFlavCvB",
                                       "Jet_btagDeepFlavCvL"});

    auto dfCuts=df.Filter("nMuon>=1");
    dfCuts=dfCuts.Filter("nJet>=4");
    dfCuts=dfCuts.Define("MuonMask","Muon_looseId && Muon_pfIsoId>1");
    dfCuts=dfCuts.Define("JetMask","Jet_muonIdx1!=0 && Jet_jetId>0 && Jet_puId>0 && Jet_pt>20 && abs(Jet_eta)<4.8");

    
    for(auto &name: df.GetColumnNames()){
        if(regex_match(name, std::regex("Muon_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[MuonMask]");
        }
        if(regex_match(name, std::regex("Jet_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[JetMask]");
        }
    }
    dfCuts=dfCuts.Redefine("nMuon","Muon_pt.size()");
    dfCuts=dfCuts.Redefine("nJet","Jet_pt.size()");
    dfCuts=dfCuts.Filter("nMuon>=1 && nJet>=4");
    dfCuts=dfCuts.Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4 && Max(Jet_btagDeepFlavB)>0.2793");

    dfCuts=dfCuts.Define("MET_eta",Met_eta,{"Muon_pt",
                                            "Muon_eta",
                                            "Muon_phi",
                                            "MET_pt",
                                            "MET_phi"});


    dfCuts=dfCuts.Define("Mu4V","ROOT::Math::PtEtaPhiMVector(Muon_pt[0],Muon_eta[0],Muon_phi[0],0.105)");
    dfCuts=dfCuts.Define("Nu4V","ROOT::Math::PtEtaPhiMVector(MET_pt,MET_eta,MET_phi,0.)");
    dfCuts=dfCuts.Define("MET_WLeptMass","(Mu4V+Nu4V).M()");
    dfCuts=dfCuts.Define("Jet4V","Jet_4Vector(Jet_pt,Jet_eta,Jet_phi,Jet_mass)");
    dfCuts=dfCuts.Define("Jet_TLeptMass",TMass,{"Mu4V",
                                            "Nu4V",
                                            "Jet4V"});
    dfCuts=dfCuts.Define("Jet_THadMass","THad_mass(Jet4V)");
    dfCuts=dfCuts.Define("Jet_WHadMass","WHad_mass(Jet4V)");                                      
    dfCuts.Snapshot("Events",output,
                              {"LHEPart_pdgId",
                               "Muon_pt",
                               "Muon_eta",
                               "Muon_phi",
                               "MET_pt",
                               "MET_eta",
                               "MET_phi",
                               "MET_WLeptMass",
                               "Jet_pt",
                               "Jet_eta",
                               "Jet_phi",
                               "Jet_mass",
                               "Jet_WHadMass",
                               "Jet_THadMass",
                               "Jet_TLeptMass",
                               "Jet_btagDeepFlavB",
                               "Jet_btagDeepFlavCvB",
                               "Jet_btagDeepFlavCvL"});

                                            

}
