def list2updown(syst_list):
    res=[]
    for syst in syst_list:
        if syst!="nominal":
            res.append(syst+"Up")
            res.append(syst+"Down")
    return res

