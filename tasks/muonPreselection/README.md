> ## TODO
> 
> - Discard the events below the trigger cuts and select only the events $W\to\mu$
> 
> - Study the acceptance for loose, medium and tight cuts (both on identification and isolation)
> 
> - Remove jets that contains the leading muon and impose the cut jet_jetid>0 e jet_puid>0
> 
> - After the cleaning of the jets, order the remaining and take the first 4
> 
> - Compute the acceptance for a cut of 20 and 30 GeV  on the fourth jet
> 
> ## Info
> 
> - [MuonID RunII](https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2)

# Muons

## Main cuts

#### Muon from W cut

- (LHEPart_pdgId[3]==-13 || LHEPart_pdgId[6]==13)

#### Trigger cuts

- abs(Muon_eta[0])<2.4

- Muon_pt[0]>26

**Total triggered leading muons:** 186533 (184272 for background)

---

## Identification and Isolation

| [signal] | Cuts                                    | Events | Fraction (over triggered muons) |
|:--------:|:---------------------------------------:| ------ |:-------------------------------:|
| Tight    | (Muon_thightId[0] && Muon_pfIsoId[0]>3) | 149597 | 0.80                            |
| Medium   | (Muon_mediumId[0] && Muon_pfIsoId[0]>2) | 156505 | 0.84                            |
| Loose    | (Muon_looseId[0] && Muon_pfIsoId[0]>1)  | 160427 | 0.86                            |

| [background] | Cuts                                    | Events | Fraction (over triggered muons) |
|:------------:|:---------------------------------------:| ------ |:-------------------------------:|
| Tight        | (Muon_thightId[0] && Muon_pfIsoId[0]>3) | 147781 | 0.80                            |
| Medium       | (Muon_mediumId[0] && Muon_pfIsoId[0]>2) | 154455 | 0.83                            |
| Loose        | (Muon_looseId[0] && Muon_pfIsoId[0]>1)  | 147781 | 0.86                            |

From now we will consider only loose muons

---

# Jets

### Cuts

- jet_jetid>0 e jet_puid>0

- Jet_muonIdx1!=0

| [signal] | Cut on fourth jet $p_t$ | Events | Fraction (over loose muons) |
|:--------:|:-----------------------:| ------ |:---------------------------:|
|          | 20 GeV                  | 139631 | 0.87                        |
|          | 30 GeV                  | 90745  | 0.56                        |

| [background] | Cut on fourth jet $p_t$ | Events | Fraction (over loose muons) |
|:------------:|:-----------------------:| ------ |:---------------------------:|
|              | 20 GeV                  | 139741 | 0.88                        |
|              | 30 GeV                  | 93562  | 0.59                        |

# Results

In the end, after all this cuts and the remotion of the muon, we choosed to impose a cut on the Leading Jet_btagDeepFlavB

| Jet_btagDeepFlavB | cut  | Signal acceptance | Background acceptance |
| ----------------- | ---- | ----------------- |:---------------------:|
| Leading           | >0.1 | 0.96              | 0.9                   |

On the subleading Jet_btagDeepFlavB we choosed to not impose a cut because the efficiencies drop instantly



--- 
