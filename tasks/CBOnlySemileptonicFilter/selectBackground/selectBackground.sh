root -b 'balanceLeptons.cpp("leptonFromTbar.root","leptonFromTbar")'
hadd -fk leptonFromTbar_new.root leptonFromTbar.root_new_e leptonFromTbar.root_new_mu leptonFromTbar.root_new_tau
root -b 'RemoveEvents.cpp("leptonFromTbar.root")'

hadd -fk TTbarSemileptonic_Nocb_LeptFromTbar.root leptonFromTbar.root leptonFromTbar_new.root


root -b 'balanceLeptons.cpp("leptonFromT.root","leptonFromT")'
hadd -fk leptonFromT_new.root leptonFromT.root_new_e leptonFromT.root_new_mu leptonFromT.root_new_tau
root -b 'RemoveEvents.cpp("leptonFromT.root")'

hadd -fk TTbarSemileptonic_Nocb_LeptFromT.root leptonFromT.root leptonFromT_new.root


hadd -fk TTbarSemileptonic_Nocb.root TTbarSemileptonic_Nocb_LeptFromTbar.root TTbarSemileptonic_Nocb_LeptFromT.root
