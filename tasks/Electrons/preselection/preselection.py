# %%

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import sys
sys.path.append("../../../utils/coffea_utils")
from coffea_utils import Electron_cuts, Jet_parton_matching



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


    selector = Electron_cuts()

    print("##############################Signal#############################")
    new_signal = selector.process(signal,LHELepton=11)
    matched_signal = Jet_parton_matching(new_signal)

    print("-----------------------------------------------------------------")
    print("############################Background###########################")

    print("Background:")
    new_background = selector.process(background,LHELepton=11)
    
    
