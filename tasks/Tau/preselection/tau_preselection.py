# %%

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
sys.path.append("../../../utils/coffea_utils")
from coffea_utils import Electron_cuts, Muon_cuts, Jet_parton_matching


# %%
if __name__ == "__main__":
    
    signal = NanoEventsFactory.from_root(
        "../TTbarSemileptonic_cbOnly_pruned_optimized.root",
        schemaclass=NanoAODSchema,
    ).events()

    background = NanoEventsFactory.from_root(
        "../TTbarSemileptonic_Nocb.root",
        schemaclass=NanoAODSchema,
    ).events()

    
    Electron_selector = Electron_cuts()
    Muon_selector = Muon_cuts()

    print("##############################Signal#############################")
    print("--------------------------Electron cuts--------------------------")
    e_signal = Electron_selector.process(signal,LHELepton=15)
    print("-----------------------------Muon cuts---------------------------")
    mu_signal = Muon_selector.process(signal,LHELepton=15)

    print("-----------------------------------------------------------------\n\n\n")
    print("############################Background###########################")

    print("--------------------------Electron cuts--------------------------")
    e_background = Electron_selector.process(background,LHELepton=15)
    print("-----------------------------Muon cuts---------------------------")
    mu_background = Muon_selector.process(background,LHELepton=15)

# %%
