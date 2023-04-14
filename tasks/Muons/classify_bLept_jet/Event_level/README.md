# Dataset

BigMuons_MuonSelection.root

After the preselection, the following cuts were applied:

- dR LHE-Jet<0.4
  
  Efficiency: 0.43

- Partons that match to the same jet => Event removed
  
  Efficiency (after dR cut): 0.917

- Just the first 7 jet in pt are considered (if one parton match a discarded jet, the event is discarded)
  
  Efficiency (after dR+multiple mathcing cut): 0.966

# JPANet

<img src="README_assets/afcc5f99b6fdd6b7e0dcd9a2077e00e5d53e862f.png" title="" alt="target.png" data-align="center">

The input features are the ones selected with the n-1,n+1 procedures (+ the WLept reco mass)

| Muons    | $p_t,\eta,\phi$                           |
| -------- | ----------------------------------------- |
| Neutrino | $p_t,\eta,\phi,m_{W_{reco}}$              |
| Jet      | $p_t,\eta,\phi,m_{Top_L}$,btagDeepFlavCvB |

Accuracy on events: 80%
