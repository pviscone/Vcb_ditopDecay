import ROOT
import os
import numpy as np

include_path=os.path.join(os.path.dirname(__file__),"RDF_utils.h")

n_thread=int(os.environ["ROOT_nTHREAD"])
ROOT.EnableImplicitMT(n_thread)
ROOT.gInterpreter.ProcessLine(f'#include "{include_path}"')


def lept_selection(rdf):
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
    
    return dfCuts
    
def jet_selection(df,lepton,weight_syst_list):
    #Jet mask
    dfCuts = (df.Define(f"JetMask", f"Jet_jetId>0 && Jet_puId>0 && Jet_pt>20 && abs(Jet_eta)<4.8")
            .Define(f"JetMatchingMask", f"muon_jet_matching(Jet_eta,Jet_phi,{lepton}_eta[0],{lepton}_phi[0])")
        )

    #!WEIGHTS
    #Jet define
    dfCuts=(dfCuts.Redefine(f"Jet_pt",f"pad_jet(Jet_pt[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_eta",f"pad_jet(Jet_eta[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_phi",f"pad_jet(Jet_phi[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_mass",f"pad_jet(Jet_mass[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavB",f"pad_jet(Jet_btagDeepFlavB[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavCvB",f"pad_jet(Jet_btagDeepFlavCvB[JetMask && JetMatchingMask],7)")
            .Redefine(f"Jet_btagDeepFlavCvL",f"pad_jet(Jet_btagDeepFlavCvL[JetMask && JetMatchingMask],7)")
            .Redefine(f"nJet",f"Sum(JetMask && JetMatchingMask)")
            .Redefine("btagW","btagW[JetMask && JetMatchingMask]")
            .Redefine("ctagW","ctagW[JetMask && JetMatchingMask]")
            .Define(f"TempEvWeights",f"ROOT::VecOps::Product(ctagW*btagW)")
            .Define(f"TempWeights",f"GenWeights*TempEvWeights")
            
        )
    
    
    for syst in weight_syst_list:
        if "btag" in syst:
            dfCuts=(dfCuts.Redefine(f"btagW_{syst}",f"btagW_{syst}[JetMask && JetMatchingMask]")
                        .Define(f"TempEvWeights_{syst}",f"ROOT::VecOps::Product(btagW_{syst})")
                        .Define(f"TempWeights_{syst}",f"GenWeights*TempEvWeights_{syst}"))
        elif "ctag" in syst:
            dfCuts=(dfCuts.Redefine(f"ctagW_{syst}",f"ctagW_{syst}[JetMask && JetMatchingMask]")
                            .Define(f"TempEvWeights_{syst}",f"ROOT::VecOps::Product(ctagW_{syst})")
                            .Define(f"TempWeights_{syst}",f"GenWeights*TempEvWeights_{syst}"))
                    

    return dfCuts

def Muon_selections(rdf,dataset,syst):
    
    #Muon Cuts
    dfCuts=(rdf.Filter(f"nMuon>0",f"{dataset}_Mu_{syst}_Loose nMuon>0")
            .Filter(f"Muon_pt[0]>26 && abs(Muon_eta[0])<2.4",f"{dataset}_Mu_{syst}_Muon[0]_pt>26 && abs(eta)<2.4"))
    
    dfCuts=(dfCuts.Filter(f"nJet>=4",f"{dataset}_Mu_{syst}_Clean nJet>=4")
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
    
    #report=dfCuts.Report()
    #report.Print()
    
    return dfCuts

    
def Electron_selections(rdf,dataset,syst):
    
    #Electron Cuts
    dfCuts=(rdf.Filter(f"nElectron>0",f"{dataset}_Ele_{syst}_Loose nElectron>0")
            .Filter(f"Electron_pt[0]>30 && abs(Electron_eta[0])<2.4",f"{dataset}_Ele_{syst}_Electron[0]_pt>30 && abs(eta)<2.4"))
    
    dfCuts=(dfCuts.Filter(f"nJet>=4",f"{dataset}_Ele_{syst}_Clean nJet>=4")
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
    
    #report=dfCuts.Report()
    #report.Print()
    
    return dfCuts

def btag_cuts(rdf):
    dfCuts=(rdf.Filter(f"Max(Jet_btagDeepFlavB)>0.2793",f"Max(Jet_btagDeepFlavB)>0.2793 (Medium)")
        )
    return dfCuts


import copy
def loop_cuts(rdf_list,cuts_func,*args):
    rdf_list=copy.copy(rdf_list)
    for i in range(len(rdf_list)):
        rdf_list[i]=cuts_func(rdf_list[i],*args)
    return rdf_list



region_cut_dict={"Muons":Muon_selections,
                "Electrons":Electron_selections}

def normalize_weights(dfCuts,
                      syst=None,
                      weight_syst_list=None,
                      sum_preWeights=None,
                      flav=None,
                      just_weights=False):
    
    assert flav in ["btag","ctag"]
    wname=f"{flav}W"
    
    #!Normalize btag weights before btag selection
    count=np.array([df.Count().GetValue() for df in dfCuts])
    if not just_weights:
        mu_w=np.array([df.Mean(f"{wname}").GetValue() for df in dfCuts])
        mu_w=sum(mu_w*count)/sum(count)
        for i in range(len(dfCuts)):
            dfCuts[i]=(dfCuts[i].Redefine(f"{wname}",f"{wname}/{mu_w}"))
        sum_preWeights=sum_preWeights/mu_w
    if syst=="nominal":
        for weight_syst in weight_syst_list:
            if flav in weight_syst:
                mu_wsyst=np.array([df.Mean(f"{wname}_{weight_syst}").GetValue() for df in dfCuts])
                mu_wsyst=sum(mu_wsyst*count)/sum(count)
                for i in range(len(dfCuts)):
                    dfCuts[i]=(dfCuts[i].Redefine(f"{wname}_{weight_syst}",f"{wname}_{weight_syst}/{mu_wsyst}"))
    if not just_weights:
        return dfCuts,sum_preWeights
    else:
        return dfCuts

def Cut(rdf_dict,
        region=None,
        dataset=None,
        syst=None,
        weight_syst_list=None,
        eff_dict=None):
    
    rdf=copy.copy(rdf_dict[dataset][syst])
    dfCuts=loop_cuts(rdf,lept_selection)
    
    if syst!="nominal":
        weight_syst_list=[]
        
    
    dfCuts=loop_cuts(dfCuts,jet_selection,region.split("s")[0],weight_syst_list)
    
    sum_preWeights=sum([df.Sum("TempWeights").GetValue() for df in dfCuts])

    dfCuts=loop_cuts(dfCuts,region_cut_dict[region],dataset,syst)

    #!normalize btag weight before btag selection
    dfCuts,sum_preWeights=normalize_weights(dfCuts,
                                            syst=syst,
                                            weight_syst_list=weight_syst_list,
                                            sum_preWeights=sum_preWeights,
                                            flav="btag")
    dfCuts=loop_cuts(dfCuts,btag_cuts)
    
    
    #!normalize ctag weight after btag selection
    dfCuts=normalize_weights(dfCuts,
                            syst=syst,
                            weight_syst_list=weight_syst_list,
                            sum_preWeights=sum_preWeights,
                            flav="ctag",
                            just_weights=True)
    
    #!Define event weights
    for i in range(len(dfCuts)):
        dfCuts[i]=(dfCuts[i].Define("Weights",f"GenWeights*ROOT::VecOps::Product(btagW*ctagW)"))
        
    if syst=="nominal":
        for weight_syst in weight_syst_list:
            for i in range(len(dfCuts)):
                if "btag" in weight_syst:
                    dfCuts[i]=(dfCuts[i].Define(f"Weights_{weight_syst}",f"GenWeights*ROOT::VecOps::Product(btagW_{weight_syst})"))
                elif "ctag" in weight_syst:
                    dfCuts[i]=(dfCuts[i].Define(f"Weights_{weight_syst}",f"GenWeights*ROOT::VecOps::Product(ctagW_{weight_syst})"))
    
    
    sum_weights=sum([df.Sum("Weights").GetValue() for df in dfCuts])

    eff=sum_weights/sum_preWeights
    eff_dict[region][dataset][syst]=eff
    
    print(f"Sum preWeights: {sum_preWeights:.2f}",flush=True)
    print(f"Sum postWeights: {sum_weights:.2f}",flush=True)
    print(f"Total efficiency: {eff*100:.2f}%\n",flush=True)
    
    if syst=="nominal":
        for weight_syst in weight_syst_list:
            
            sum_systWeights=sum([df.Sum(f"Weights_{weight_syst}").GetValue() for df in dfCuts])
            
            eff=sum_systWeights/sum_preWeights
            eff_dict[region][dataset][weight_syst]=eff
            print(f"{weight_syst} efficiency: {eff*100:.2f}%",flush=True)

    return dfCuts,eff_dict
    


