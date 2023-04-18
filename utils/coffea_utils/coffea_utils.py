import awkward as ak
from coffea import processor
from coffea.analysis_tools import PackedSelection
import numpy as np


def np_and(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.bitwise_and(arg, res)
    return res

def np_or(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.bitwise_or(arg, res)
    return res

def print_cuts(selection):
    cut_set = set()
    n_events=selection.all(*cut_set).sum()
    for cut in selection.names:
        num_before_cut = selection.all(*cut_set).sum()
        cut_set.add(cut)
        num_after_cut = selection.all(*cut_set).sum()
        print(
            f"Cut: {cut}\n\n"
                +" "*15+f"Relative efficiency: {(num_after_cut/num_before_cut):.3f} "
                +f"+- {(np.sqrt(num_after_cut)/num_before_cut):.3f}\n"
                +" "*15+f"Cumulative efficiency: {(num_after_cut/n_events):.3f} "
                +f"+- {(np.sqrt(num_after_cut)/n_events):.3f}\n"
        )
        print("-----------------------------------------------------------------")
    return cut_set


def MET_eta(lepton,MET):
    lept_pt=lepton.pt.to_numpy()
    lept_eta=lepton.eta.to_numpy()
    lept_phi=lepton.phi.to_numpy()
    MET_pt=MET.pt.to_numpy()
    MET_phi=MET.phi.to_numpy()
    Mw = 80.385
    El2 = lept_pt**2*np.cosh(lept_eta)**2
    Pt_scalar_product = MET_pt*lept_pt*np.cos(MET_phi-lept_phi)
    a = lept_pt**2
    b = -lept_pt*np.sinh(lept_eta)*(Mw**2+2*Pt_scalar_product)
    c = (-(Mw**2+2*Pt_scalar_product)**2+4*El2*(MET_pt**2))/4
    delta = b**2-4*a*c
    mask = delta < 0
    delta[mask] = 0
    res = ((-b-np.sqrt(delta))/(2*a))
    res=np.arcsinh(res/MET_pt)
    return ak.Array(res)

def Jet_parton_matching(events,num_jet_to_select=7):
    
    #!-------------------Selecting the objects-------------------!
    #* LHE Mask
    #Select only the first product of the W decay
    pdgId_Wdecay= events.LHEPart.pdgId[:,[3,6]]
    #Mask for the leptonic decay of the W
    leptonic_LHE_mask=np_or(pdgId_Wdecay==13,
                            pdgId_Wdecay==-13,
                            pdgId_Wdecay==11,
                            pdgId_Wdecay==-11,
                            pdgId_Wdecay==15,
                            pdgId_Wdecay==-15)
    hadronic_LHE_mask=np.bitwise_not(leptonic_LHE_mask)

    #*Define Jet objects
    near_Jet,deltaR = events.LHEPart.nearest(events.Jet,return_metric=True)
    bLept_Jet = near_Jet[:, [2, 5]][leptonic_LHE_mask]
    bHad_Jet = near_Jet[:, [2, 5]][hadronic_LHE_mask]
    Wb_Jet= near_Jet[:, [4, 6]][hadronic_LHE_mask]
    Wc_Jet= near_Jet[:, [3, 7]][hadronic_LHE_mask]

    #*Compute the deltaR between the LHE and the Jet
    deltaRLept=deltaR[:, [2, 5]][leptonic_LHE_mask]
    deltaRHad=deltaR[:, [2, 5]][hadronic_LHE_mask]
    deltaRWb=deltaR[:, [4, 6]][hadronic_LHE_mask]
    deltaRWc=deltaR[:, [3, 7]][hadronic_LHE_mask]
    
    del hadronic_LHE_mask, leptonic_LHE_mask, deltaR, near_Jet, pdgId_Wdecay

    #*Compute the index in pt of the selected jets
    bLept_pt_order = ak.argmax(events.Jet.pt==bLept_Jet.pt, axis=1)
    bHad_pt_order = ak.argmax(events.Jet.pt==bHad_Jet.pt, axis=1)
    Wb_pt_order = ak.argmax(events.Jet.pt==Wb_Jet.pt, axis=1)
    Wc_pt_order = ak.argmax(events.Jet.pt==Wc_Jet.pt, axis=1)


    #%%
    # * Apply the mask
    Rmask = np_and(deltaRLept < 0.4,
                deltaRHad < 0.4,
                deltaRWb < 0.4,
                deltaRWc < 0.4,
                )


    print(f"dR<0.4 efficiency: {np.sum(Rmask)/len(Rmask):.2f}")

    #*Compute the order of the jets in pt and the efficiency of selecting the first N

    same_match_matrix = np.array([
        (bLept_pt_order == bHad_pt_order),
        (bLept_pt_order == Wb_pt_order),
        (bLept_pt_order == Wc_pt_order),
        (bHad_pt_order == Wb_pt_order),
        (bHad_pt_order == Wc_pt_order),
        (Wb_pt_order == Wc_pt_order)]).T

    same_match_event_mask = np.bitwise_or.reduce(same_match_matrix, axis=1)


    print("Multiple match to the same jet"
          +f" (After RMask): {np.sum(same_match_event_mask[Rmask])/np.sum(Rmask):.2%}")


    #!-------------------Masking-------------------!
    R_multiple_match_mask = np.bitwise_and(Rmask, np.bitwise_not(same_match_event_mask))

    mask = np_and(Rmask,
                np.bitwise_not(same_match_event_mask),
                bLept_pt_order < num_jet_to_select,
                bHad_pt_order < num_jet_to_select,
                Wb_pt_order < num_jet_to_select,
                Wc_pt_order < num_jet_to_select,)

    efficiency = ak.sum(mask)/ak.sum(R_multiple_match_mask)
    print(
        "Percentage of events with selected jets in the first"
        +f" {num_jet_to_select} jets:{efficiency*100:.2f}%")
    return events[mask]



class Electron_cuts(processor.ProcessorABC):
    def __init__(self):
        pass

    def postprocess(self, accumulator):
        pass

    def process(self, events, LHELepton=None):
        if LHELepton is not None:
            assert LHELepton in [11,13,15]
            lepton_LHEMask = np.bitwise_or(
                np.abs(events.LHEPart.pdgId[:, 3]) == LHELepton,
                np.abs(events.LHEPart.pdgId[:, 6]) == LHELepton,
            )
            events = events[lepton_LHEMask]

        events["Electron"] = ak.pad_none(events.Electron, 1)
        events["Jet"] = events.Jet[
            np_and(
                events.Jet.electronIdx1 != 0,
                events.Jet.jetId > 0,
                events.Jet.puId > 0,
                events.Jet.pt > 20,
                np.abs(events.Jet.eta) < 4.8,
            )
        ]
        

        selection = PackedSelection()
        selection.add(
            "nElectrons>=1",
            ak.count(events.Electron.pt,axis=1) >= 1)
        selection.add(
            "Electron trigger(pt[0]>30 && |eta|[0]<2.4)",
            np_and(events.Electron.pt[:, 0] > 30,
                   np.abs(events.Electron.eta[:, 0]) < 2.4))
        selection.add(
            "mvaFall17V2Iso_WP90",
            events.Electron.mvaFall17V2Iso_WP90[:, 0])
        selection.add(
            "nJet(electronIdx1!=0 & jetId>0 & puId>0 & pt>20 & |eta|<4.8)>=4",
            ak.count(events.Jet.pt,axis=1) >= 4)
        selection.add(
            "max(DeepFlavB) medium 0.2793",
            ak.max(events.Jet.btagDeepFlavB, axis=1) > 0.2793)

        cut_set=print_cuts(selection)
        return events[selection.all(*cut_set)]


class Muon_cuts(processor.ProcessorABC):
    def __init__(self):
        pass

    def postprocess(self, accumulator):
        pass

    def process(self, events,LHELepton=None):
        if LHELepton is not None:
            assert LHELepton in [11,13,15]
            lepton_LHEMask = np.bitwise_or(
                np.abs(events.LHEPart.pdgId[:, 3]) == LHELepton,
                np.abs(events.LHEPart.pdgId[:, 6]) == LHELepton,
            )
            events = events[lepton_LHEMask]
        
        events["Muon"] = ak.pad_none(events.Muon,1)
        events["Jet"] = events.Jet[
            np_and(
                events.Jet.muonIdx1 != 0,
                events.Jet.jetId > 0,
                events.Jet.puId > 0,
                events.Jet.pt > 20,
                np.abs(events.Jet.eta) < 4.8,
            )
        ]

        selection = PackedSelection()

        selection.add(
            "nMuons>=1",
            ak.count(events.Muon.pt,axis=1) >= 1)
        selection.add(
            "Muon trigger(pt[0]>26 && |eta|[0]<2.4)",
            np_and(events.Muon.pt[:, 0] > 26,
                   np.abs(events.Muon.eta[:, 0]) < 2.4))
        selection.add(
            "looseId && pfIsoId loose (>1)",
            np_and(events.Muon.looseId[:, 0],
                   events.Muon.pfIsoId[:, 0] > 1))
        selection.add(
            "nJet(muonIdx1!=0 & jetId>0 & puId>0 & pt>20 & |eta|<4.8)>=4",
            ak.count(events.Jet.pt,axis=1) >= 4)
        selection.add(
            "max(DeepFlavB) medium 0.2793",
            ak.max(events.Jet.btagDeepFlavB, axis=1) > 0.2793)

        cut_set = print_cuts(selection)
        return events[selection.all(*cut_set)]
