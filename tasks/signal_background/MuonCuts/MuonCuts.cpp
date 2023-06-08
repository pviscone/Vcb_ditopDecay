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

RVec<float> pad_jet(const RVec<float> &jet_vec, int n){
    RVec<float> jet_vec_pad(n);
    for(int i=0;i<n;i++){
        if(i<jet_vec.size()){
            jet_vec_pad[i]=jet_vec[i];
        }else{
            jet_vec_pad[i]=0.;
        }
    }
    return jet_vec_pad;
}

RVec<float> WHad_mass(const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V){
    int n=Jet4V.size();
    RVec<float> Wmass(n*n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if (Jet4V[i].Pt()==0. or Jet4V[j].Pt()==0.){
                Wmass[i*n+j]=0.;
            }
            else{
                if (i==j){
                    Wmass[i*n+j] = (Jet4V[i]).M();
                } else{
                Wmass[i*n+j]=(Jet4V[i]+Jet4V[j]).M();
                }
            }
        }
    }
    return Wmass;
}



RVec<float> THad_mass(const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V){
    int n=Jet4V.size();
    RVec<float> Tmass(n*n*n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            for(int k=0;k<n;k++){
                if (Jet4V[i].Pt()==0. or Jet4V[j].Pt()==0. or Jet4V[k].Pt()==0.){
                    Tmass[i*n*n+n*j+k]=0.;
                }
                else{
                    if (i==j and j==k){
                        Tmass[i*n*n + n*j + k] = (Jet4V[i]).M();
                    } else{
                        Tmass[i*n*n + n*j + k] = (Jet4V[i] + Jet4V[j] + Jet4V[k]).M();
                    }
                }
            }
        }
    }
    return Tmass;
}

RVec<float> Masses(const RVec<ROOT::Math::PtEtaPhiMVector> &Jet4V,
                   const ROOT::Math::PtEtaPhiMVector &Mu4V,
                   const ROOT::Math::PtEtaPhiMVector &Nu4V){
    int n=Jet4V.size()+1;
    auto W4V=Mu4V+Nu4V;
    RVec<float> masses(n*(n+1)/2);
    int idx=0;
    for (int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i>j){
                continue;
            }
            if(i==0 and j==0){
                masses[idx]=(W4V).M();
            } else if (i==0){
                if (Jet4V[j-1].M()==0.){
                    masses[idx]=0.;
                } else{
                    masses[idx]=(W4V+Jet4V[j-1]).M();
                }
            } else if (j==0){
                if (Jet4V[i-1].M()==0.){
                    masses[idx]=0.;
                } else{
                    masses[idx]=(W4V+Jet4V[i-1]).M();
                }
            } else if(i==j){
                masses[idx]=Jet4V[i-1].M();
            } else{
                if (Jet4V[i-1].M()==0. or Jet4V[j-1].M()==0.){
                    masses[idx]=0.;
                } else{
                    masses[idx]=(Jet4V[i-1]+Jet4V[j-1]).M();
                }
            }
            idx++;
        }
    }
    return masses;
}

RVec<bool> muon_jet_matching(const RVec<float> &eta_jet,
                             const RVec<float> &phi_jet,
                             const float &eta_muon,
                             const float &phi_muon){
    RVec<float>phi_muon_vec(eta_jet.size(),phi_muon);
    RVec<float>eta_muon_vec(eta_jet.size(),eta_muon);
    RVec<float>deltaR=ROOT::VecOps::DeltaR(eta_jet,eta_muon_vec,phi_jet,phi_muon_vec);
    return deltaR>0.4;
}

RVec<float> SecondLepton(const RVec<float> &Muon, const RVec<float> &Electron){
    RVec<float> res(3);
    if (Muon.size()<2){
        res[0]=0;
    }else{
        res[0]=Muon[1];
    }

    int n_electron= Electron.size();
    if(n_electron<1){
        res[1]=0;
        res[2]=0;
    }
    else if (n_electron<2){
        res[1]=Electron[0];
        res[2]=0;
    }
    else{
        res[1]=Electron[0];
        res[2]=Electron[1];
    }
    return res;
}



    //"./powheg/root_files"
void MuonCuts(std::string input, std::string output) {
    std::vector<std::string> files_path;
    if(input.find(".root")==string::npos){
        for (const auto & entry : std::filesystem::directory_iterator(input)){
            files_path.push_back(entry.path());
        }
    }else{
        files_path.push_back(input);
    }
    EnableImplicitMT();
    
    RDataFrame df("Events",files_path,{"LHEPart_pdgId",
                                       "LHEPart_pt",
                                       "LHEPart_eta",
                                       "LHEPart_phi",
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
                                       "Jet_area",
                                       "Jet_muonIdx1",
                                       "Jet_jetId",
                                       "Jet_puId",
                                       "nJet",
                                       "Jet_btagDeepFlavB",
                                       "Jet_btagDeepFlavCvB",
                                       "Jet_btagDeepFlavCvL"});

    auto dfCuts=df.Filter("nMuon>=0","nEvents");

    dfCuts=dfCuts.Define("MuonMask","Muon_looseId && Muon_pfIsoId>1");
                 

    
    for(auto &name: df.GetColumnNames()){
        if(regex_match(name, std::regex("Muon_.*"))){
            dfCuts = dfCuts.Redefine(name, name+"[MuonMask]");
        }

    }

    dfCuts = dfCuts.Define("JetMask", "Jet_jetId>0 && Jet_puId>0 && Jet_pt>20 && abs(Jet_eta)<4.8")
                   .Define("JetMatchingMask", "muon_jet_matching(Jet_eta,Jet_phi,Muon_eta[0],Muon_phi[0])");

    for (auto &name : df.GetColumnNames()) {
        if (regex_match(name, std::regex("Jet_.*"))) {
            dfCuts = dfCuts.Redefine(name, name + "[JetMask && JetMatchingMask]");
        }
    }

    dfCuts = dfCuts.Redefine("nMuon", "Muon_pt.size()")
                    .Redefine("nJet", "Jet_pt.size()")
                    .Filter("nMuon>=1", "Loose nMuon >=1")
                    .Filter("Muon_pt[0]>26 && abs(Muon_eta[0])<2.4", "Muon[0] pt>26 && abs(eta)<2.4")
                    .Filter("nJet>=4", "Clean nJet>=4")
                    .Filter("Max(Jet_btagDeepFlavB)>0.2793", "Max DeepFlavB>0.2793 (Medium)");

    auto report = dfCuts.Report();
    report->Print();

    for (auto &name : df.GetColumnNames()) {
        if (regex_match(name, std::regex("Jet_.*"))) {
            dfCuts = dfCuts.Redefine(name, "pad_jet(" + name + ",7)");
        }
    }
    dfCuts = dfCuts.Define("MET_eta", Met_eta, {"Muon_pt", "Muon_eta", "Muon_phi", "MET_pt", "MET_phi"})
                 .Define("Mu4V", "ROOT::Math::PtEtaPhiMVector(Muon_pt[0],Muon_eta[0],Muon_phi[0],0.105)")
                 .Define("Nu4V", "ROOT::Math::PtEtaPhiMVector(MET_pt,MET_eta,MET_phi,0.)")
                 .Define("MET_WLeptMass", "(Mu4V+Nu4V).M()")
                 .Define("Jet4V", "Jet_4Vector(Jet_pt,Jet_eta,Jet_phi,Jet_mass)")
                 .Define("Masses", Masses, {"Jet4V", "Mu4V", "Nu4V"})
                 .Define("SecondLept_pt", SecondLepton, {"Muon_pt", "Electron_pt"})
                 .Define("SecondLept_eta", SecondLepton, {"Muon_eta", "Electron_eta"})
                 .Define("SecondLept_phi", SecondLepton, {"Muon_phi", "Electron_phi"});

    dfCuts.Snapshot("Events", output,
                    {"LHEPart_pdgId",
                     "Muon_pt",
                     "Muon_eta",
                     "Muon_phi",
                     "MET_pt",
                     "MET_eta",
                     "MET_phi",
                     "Jet_pt",
                     "Jet_eta",
                     "Jet_phi",
                     "Jet_area",
                     "Jet_btagDeepFlavB",
                     "Jet_btagDeepFlavCvB",
                     "Jet_btagDeepFlavCvL",
                     "Masses",
                     "SecondLept_pt",
                     "SecondLept_eta",
                     "SecondLept_phi"});
    exit(0);
}
