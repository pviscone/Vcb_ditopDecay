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

<img title="" src="README_assets/c71f1edb55db47efa23737171eedcce1ee0f5a09.png" alt="target2.png" data-align="left" width="711">

The input features are the ones selected with the n-1,n+1 procedures (+ the WLept reco mass and btagDeepFlavCvL)

| Muons    | $p_t,\eta,\phi$                                           |
| -------- | --------------------------------------------------------- |
| Neutrino | $p_t,\eta,\phi,m_{W_{reco}}$                              |
| Jet      | $p_t,\eta,\phi,m_{Top_L}$,btagDeepFlavCvL,btagDeepFlavCvB |

For each event, the output is a matrix 7 x 4 (jet x classes). (The probabilities sum up to 1 along the jets)

Taking all the 7!/(7-4)! combination we take the one which has the higher product of the jet scores

Accuracy on events: 54%
