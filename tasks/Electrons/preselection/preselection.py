# %%
import mplhep
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection


import numpy as np
import matplotlib.pyplot as plt

signal = NanoEventsFactory.from_root(
    "../TTbarSemileptonic_cbOnly_pruned_optimized.root",
    schemaclass=NanoAODSchema,
).events()

signal_eLHEMask = np.bitwise_or(
    np.abs(signal.LHEPart.pdgId[:, 3]) == 11,
    np.abs(signal.LHEPart.pdgId[:, 6]) == 11,
)

signal=signal[signal_eLHEMask]

background = NanoEventsFactory.from_root(
    "../TTbarSemileptonic_Nocb.root",
    schemaclass=NanoAODSchema,
).events()

background_eLHEMask = np.bitwise_or(
    np.abs(background.LHEPart.pdgId[:, 3]) == 11,
    np.abs(background.LHEPart.pdgId[:, 6]) == 11,
)
background=background[background_eLHEMask]

# %%



class selections(processor.ProcessorABC):
    def __init__(self):
        pass

    def postprocess(self, accumulator):
        pass

    def process(self, events):
        electron_LHEMask = np.bitwise_or(
            np.abs(events.LHEPart.pdgId[:, 3]) == 11,
            np.abs(events.LHEPart.pdgId[:, 6]) == 11,
        )
        events = events[electron_LHEMask]

        events.Jet = events.Jet[events.Jet.electronIdx1 != 0]
        events.Jet = events.Jet[events.Jet.jetId > 0]
        events.Jet = events.Jet[events.Jet.puId > 0]

        selection = PackedSelection()
        
        nElectron_geq1_mask = ak.count(events.Electron.pt, axis=1) >= 1
        Electron_trigger_mask = np.zeros(len(nElectron_geq1_mask), dtype=bool)
        Electron_trigger_mask[nElectron_geq1_mask] = np.bitwise_and(
            events.Electron.pt[nElectron_geq1_mask, 0] > 30,
            np.abs(events.Electron.eta[nElectron_geq1_mask, 0]) < 2.4
            )
        Elecron_Iso_mask = np.zeros(len(nElectron_geq1_mask), dtype=bool)
        Elecron_Iso_mask[nElectron_geq1_mask] = \
                            events.Electron.mvaFall17V2Iso_WP90[nElectron_geq1_mask, 0]
        
        nJet_geq4_mask = ak.count(events.Jet.pt, axis=1) >= 4
        Jet_4pt_mask = np.zeros(len(nJet_geq4_mask), dtype=bool)
        Jet_4pt_mask[nJet_geq4_mask] = events.Jet[nJet_geq4_mask].pt[:, 3] > 20
            
        selection.add("nElectrons>=1",
                        nElectron_geq1_mask)
        selection.add("Electron trigger(pt[0]>30 && |eta|[0]<2.4)",
                        Electron_trigger_mask)
        selection.add("mvaFall17V2Iso_WP90",
                        Elecron_Iso_mask)
        selection.add("nJets(electronIdx1!=0 && jetId>0 && puId>0)>=4",
                        nJet_geq4_mask)
        selection.add("Jet[3].pt>20",
                        Jet_4pt_mask)
        selection.add("max(DeepFlavB) medium 0.2793",
                        ak.max(events.Jet.btagDeepFlavB, axis=1) > 0.2793)

        cut_set = set()
        for cut in selection.names:
            num_before_cut = selection.all(*cut_set).sum()
            cut_set.add(cut)
            num_after_cut = selection.all(*cut_set).sum()
            print(
                f"Cut: {cut} \n\n\
                    Relative efficiency: {(num_after_cut/num_before_cut):.2f}\n\
                    Cumulative efficiency: {(num_after_cut/len(events)):.2f}"
            )
            print("-----------------------------------------------------------------")
        events = events[selection.all(*cut_set)]
        return events


# %%
selector = selections()

print("##############################Signal#############################")
new_signal = selector.process(signal)

print("-----------------------------------------------------------------")
print("############################Background###########################")

print("Background:")
new_background = selector.process(background)
