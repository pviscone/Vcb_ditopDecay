# Cuts summary

(The efficiency for signal and background is different only if specified)

The efficiency reported in each subsection refers after the cuts of the previous subsection

## Muons

At least 1 $\mu$: Acceptance: 0.954

###### Trigger cuts:

- abs(Muon_eta[0])<2.4 && Muon_pt[0]>26

Acceptance: 0.756

###### Identification and isolation

- (Muon_looseId[0] && Muon_pfIsoId[0]>1)

Acceptance: 0.86

## Jets

###### Selected jets

- jet_jetid>0 && jet_puid>0 && Jet_muonIdx1!=0

###### P_t cuts

- Jet_pt[3]>20 && nJet (after selected jets)>=4

Acceptance: 0.84

## btag

- max(Jet_btagDeepFlavB) > 0.2793 (medium)

Acceptance:

- Signal : 0.968

- Background: 0.921 

---

## Jet matching

For the signal, after the cuts discussed above:

- $\Delta R_{LHE-JET}<0.4$
  
  Acceptance: 0.43

- Different Partons match to different Jets 
  
  Acceptance: 0.917

- All the 4 partons are in the firsts 7 jets in pt  
  
  Acceptance: 0.966

---

# Classification

JPANet was the best classifier

###### Jet Parton Assignment

|         | Efficiency |
| ------- | ---------- |
| bLept   | 0.80       |
| allJets | 0.54       |
