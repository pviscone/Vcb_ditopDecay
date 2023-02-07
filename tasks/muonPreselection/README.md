(prendi il dizionario del nanoaod che stai usando)

Fai i seguienti tagli:

- accettanza eta: abs(Muon_eta)<2.4
- accettanza pt: Muon_pt[0]>26
- Seleziona solo eventi con un mu dal W (LHEPart_pdgId[3]==-13 ||  LHEPart_pdgId[6]==13)

Vedi quanti eventi perdi prendendo muoni loose (Muon_looseId[0] && Muon_pfIsoId[0]>1)

---

Nei jet c'è anche il muone. Rimuovi il muone dalla collezione di jet, ordina i primi 4 restanti e vedi gli spettri del pt

Fissa anche i tagli jet_jetid>0 e jet_puid>0

Calcola quanti eventi perdi con un taglio sul quarto jet a 20 e 30 gev

--- 

Leggi twiki su oggetti muoni e vedi anche quelli sui jet cosa sono
