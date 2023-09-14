import pprint
import numpy as np


lumi = 138e3
score_dict = {}
score_dict["Muons"] = {
    "signalMuons": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "signalElectrons": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "signalTaus": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "bkg": lumi * 832 * 0.44 * (1 - 8.4e-4),  # bkg no efficiencies
    "diHad": lumi * 832 * 0.45,
    "diLept": lumi * 832 * 0.11,
    "WJets": lumi * 59100 * 0.108 * 3,
}

score_dict["Electrons"] = {
    "signalElectrons": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "signalMuons": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "signalTaus": lumi * 832 * 0.44 * 0.33 * 8.4e-4,
    "bkg": lumi * 832 * 0.44 * (1 - 8.4e-4),  # bkg no efficiencies
    "diHad": lumi * 832 * 0.45,
    "diLept": lumi * 832 * 0.11,
    "WJets": lumi * 59100 * 0.108 * 3,
}
pprint.pprint(score_dict)
# eff

# Semilept  Electrons 0.156 Muons 0.193
# TauSemi    Electrons 0.0095 Muons 0.0136
