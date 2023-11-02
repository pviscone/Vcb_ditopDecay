import numpy as np
import awkward as ak

def np_and(*args):
    res=args[0]
    for i in args[1:]:
        res=np.bitwise_and(res,i)
    return res

def np_or(*args):
    res=args[0]
    for i in args[1:]:
        res=np.bitwise_or(res,i)
    return res

def MET_eta(lepton,MET):
    lept_pt=lepton.pt.to_numpy()
    lept_eta=lepton.eta.to_numpy()
    lept_phi=lepton.phi.to_numpy()
    MET_pt=MET.pt.to_numpy()
    MET_phi=MET.phi.to_numpy()
    Mw = 80.385
    El2 = lept_pt**2*np.cosh(lept_eta)**2
    Pt_scalar_product = MET_pt*lept_pt*np.cos(MET_phi-lept_phi)
    a = lept_pt**2
    b = -lept_pt*np.sinh(lept_eta)*(Mw**2+2*Pt_scalar_product)
    c = (-(Mw**2+2*Pt_scalar_product)**2+4*El2*(MET_pt**2))/4
    delta = b**2-4*a*c
    mask = delta < 0
    delta[mask] = 0
    res = ((-b-np.sqrt(delta))/(2*a))
    res=np.arcsinh(res/MET_pt)
    return ak.Array(res)

def obj_selection(events):
    #Muons
    events["Muon"]=events.Muon[np_and(events.Muon.pt>26,
                                      events.Muon.looseId,
                                      events.Muon.pfIsoId > 1,
                                      np.abs(events.Muon.eta)<2.4)]

    
    #Jets
    events["Jet"]=events.Jet[np_and(events.Jet.pt>20,
                                    np.abs(events.Jet.eta)<4.8,
                                    events.Jet.jetId>0,
                                    events.Jet.puId > 0)]
    
    lept=ak.concatenate([events.Electron,events.Muon],axis=1)
    lept_argsort=ak.argsort(lept.pt,axis=1,ascending=False)
    lept=lept[lept_argsort]
    lept=ak.pad_none(lept,1,clip=True)
    jet_lept_dR=events.Jet.delta_r(lept)
    jet_lept_dR=ak.fill_none(jet_lept_dR,999)
    
    events["Jet"]=events.Jet[jet_lept_dR>0.4]
    return events



def mu_selection(events):
    events=events[ak.count(events.Muon.pt,axis=1)>=1]
    events=events[ak.count(events.Jet.pt,axis=1)>=4]
    events=events[ak.max(events.Jet.btagDeepFlavB,axis=1)>0.277]
    events["MET","eta"]=MET_eta(events.Muon[:,0],events.MET)
    events["W"]=events.Muon[:,0]+events.MET
    mu=events.Muon[:,0]
    nu=events.MET
    events["W","eta"]=np.arcsinh((nu.pt*np.sinh(nu.eta)+mu.pz)/((nu+mu).pt))
    return events[~ak.is_none(events)]


