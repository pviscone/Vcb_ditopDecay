import ROOT
import os


include_path=os.path.join(os.path.dirname(__file__),"RDF_utils.h")

ROOT.EnableImplicitMT()
ROOT.gInterpreter.ProcessLine(f'#include "{include_path}"')


def object_selection(rdf):
    #Muon and electron masks
    dfCuts=(rdf.Define(f"MuonMask",f"Muon_looseId && Muon_pfIsoId>1")
            .Define(f"ElectronMask", f"Electron_mvaFall17V2Iso_WP90")
        )

    #Muon define
    dfCuts=(dfCuts.Redefine(f"Muon_pt",f"Muon_pt[MuonMask]")
            .Redefine(f"Muon_eta",f"Muon_eta[MuonMask]")
            .Redefine(f"Muon_phi",f"Muon_phi[MuonMask]")
            .Redefine(f"nMuon",f"Muon_pt.size()")
        )

    #Electron redefine
    dfCuts=(dfCuts.Redefine(f"Electron_pt",f"Electron_pt[ElectronMask]")
            .Redefine(f"Electron_eta",f"Electron_eta[ElectronMask]")
            .Redefine(f"Electron_phi",f"Electron_phi[ElectronMask]")
            .Redefine(f"nElectron",f"Electron_pt.size()")
        )

    #Jet mask
    dfCuts = (dfCuts.Define(f"JetMask", f"Jet_jetId>0 && Jet_puId>0 && Jet_pt>20 && abs(Jet_eta)<4.8")
            .Define(f"JetMatchingMask", f"muon_jet_matching(Jet_eta,Jet_phi,Muon_eta[0],Muon_phi[0])")
        )

    #Jet define
    dfCuts=(dfCuts.Redefine(f"Jet_pt",f"pad_jet(Jet_pt[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_eta",f"pad_jet(Jet_eta[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_phi",f"pad_jet(Jet_phi[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_mass",f"pad_jet(Jet_mass[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavB",f"pad_jet(Jet_btagDeepFlavB[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavCvB",f"pad_jet(Jet_btagDeepFlavCvB[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavCvL",f"pad_jet(Jet_btagDeepFlavCvL[JetMask && JetMatchingMask],7)")
            .Redefine(f"nJet",f"Jet_pt.size()")
        )
    
    return dfCuts

def Muon_selections(rdf,dataset,syst):
    
    #Muon Cuts
    dfCuts=(rdf.Filter(f"nMuon>0",f"{dataset}_Mu_{syst}_Loose nMuon>0")
            .Filter(f"Muon_pt[0]>26 && abs(Muon_eta[0])<2.4",f"{dataset}_Mu_{syst}_Muon[0]_pt>26 && abs(eta)<2.4")
            .Filter(f"nJet>=4",f"{dataset}_Mu_{syst}_Clean nJet>=4")
            .Filter(f"Max(Jet_btagDeepFlavB)>0.2793",f"{dataset}_Mu_{syst}_Max DeepFlavB>0.2793 (Medium)")
        )
    
    #Define new objects
    dfCuts = (dfCuts.Define(f"MET_eta", f"Met_eta(Muon_pt, Muon_eta, Muon_phi, MET_pt, MET_phi)")
                 .Define(f"Mu4V", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[0],Muon_eta[0],Muon_phi[0],0.105)")
                 .Define(f"Nu4V", f"ROOT::Math::PtEtaPhiMVector(MET_pt,MET_eta,MET_phi,0.)")
                 .Define(f"MET_WLeptMass", f"(Mu4V+Nu4V).M()")
                 .Define(f"Jet4V", f"Jet_4Vector(Jet_pt,Jet_eta,Jet_phi,Jet_mass)")
                 .Define(f"Masses", f"Masses(Jet4V, Mu4V, Nu4V)")
                 .Define(f"SecondLept_pt", f"SecondLepton(Muon_pt, Electron_pt)")
                 .Define(f"SecondLept_eta", f"SecondLepton(Muon_eta, Electron_eta)")
                 .Define(f"SecondLept_phi", f"SecondLepton(Muon_phi, Electron_phi)")
        )
    
    report=dfCuts.Report()
    report.Print()
    
    return dfCuts

    
def Electron_selections(rdf,dataset,syst):
    
    #Electron Cuts
    dfCuts=(rdf.Filter(f"nElectron>0",f"{dataset}_Ele_{syst}_Loose nElectron>0")
            .Filter(f"Electron_pt[0]>30 && abs(Electron_eta[0])<2.4",f"{dataset}_Ele_{syst}_Electron[0]_pt>30 && abs(eta)<2.4")
            .Filter(f"nJet>=4",f"{dataset}_Ele_{syst}_Clean nJet>=4")
            .Filter(f"Max(Jet_btagDeepFlavB)>0.2793",f"{dataset}_Ele_{syst}_Max DeepFlavB>0.2793 (Medium)")
        )
    
    #Define new objects
    dfCuts = (dfCuts.Define(f"MET_eta", f"Met_eta(Muon_pt, Muon_eta, Muon_phi, MET_pt, MET_phi)")
                 .Define(f"E4V", f"ROOT::Math::PtEtaPhiMVector(Electron_pt[0],Electron_eta[0],Muon_phi[0],0.105)")
                 .Define(f"Nu4V", f"ROOT::Math::PtEtaPhiMVector(MET_pt,MET_eta,MET_phi,0.)")
                 .Define(f"MET_WLeptMass", f"(E4V+Nu4V).M()")
                 .Define(f"Jet4V", f"Jet_4Vector(Jet_pt,Jet_eta,Jet_phi,Jet_mass)")
                 .Define(f"Masses", f"Masses(Jet4V, E4V, Nu4V)")
                 .Define(f"SecondLept_pt", f"SecondLepton(Electron_pt, Muon_pt)")
                 .Define(f"SecondLept_eta", f"SecondLepton(Electron_eta, Muon_eta)")
                 .Define(f"SecondLept_phi", f"SecondLepton(Electron_phi, Muon_phi)")
        )
    
    report=dfCuts.Report()
    report.Print()
    
    return dfCuts


def Cuts(rdf,dataset,syst):
    dfCuts=object_selection(rdf)
    dfCuts_Muon=Muon_selections(dfCuts,dataset,syst)
    dfCuts_Electron=Electron_selections(dfCuts,dataset,syst)
    
    return {"Muons":dfCuts_Muon,"Electrons":dfCuts_Electron}
    
    


def systematics_cutloop(rdf_dict,syst_dict,):
    rdf_dict["TTsemiLept"]={}
    rdf_dict["TTsemiLept_Tau"]={}
    for systematic in (syst_dict):
        rdf_dict["TTsemiLept"][systematic]=rdf_dict["bkg"][systematic].Filter("Sum(abs(LHEPart_pdgId)==15)==0","NoTaus")
        rdf_dict["TTsemiLept_Tau"][systematic]=rdf_dict["bkg"][systematic].Filter("Sum(abs(LHEPart_pdgId)==15)>0","Taus")
    del rdf_dict["bkg"]
    

    
    rdf_MuE_dict={}
    rdf_MuE_dict["Muons"]={}
    rdf_MuE_dict["Electrons"]={}
    for dataset in (rdf_dict):
        print(f"###############{dataset}###############")
        

        rdf_MuE_dict["Muons"][dataset]={}
        rdf_MuE_dict["Electrons"][dataset]={}
        

        for systematic in (syst_dict):
            print("Systematic: ", systematic)
            print("--------------------")
            res=Cuts(rdf_dict[dataset][systematic],dataset,systematic)
            rdf_MuE_dict["Muons"][dataset][systematic]=res["Muons"]
            rdf_MuE_dict["Electrons"][dataset][systematic]=res["Electrons"]
            
    return rdf_MuE_dict
        