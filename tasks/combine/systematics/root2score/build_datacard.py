
def build_datacard(sample_dict,
                   regions=None,
                   syst_list=None,
                   autoMCStats=True):
    n=len(sample_dict)
    syst_list.remove("nominal")
    syst_list=[name.split("Up")[0].split("Down")[0] for name in syst_list]
    syst_list=list(set(syst_list))
    
    datacard=open("autodatacard.txt","w")
    datacard.write("imax 2\n")
    datacard.write("jmax *\n")
    datacard.write("kmax *\n")
    datacard.write("---------------\n")
    datacard.write("shapes * * hist.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_syst_$SYSTEMATIC\n")
    datacard.write("---------------\n")
    datacard.write("bin\tMuons\tElectrons\n\n---------------\n")
    datacard.write("bin\t")
    for i in range(len(sample_dict)):
        for region in regions:
            datacard.write(region+"\t")
    datacard.write("\nprocess\t")
    for sample in sample_dict:
        for i in range(len(regions)):
            datacard.write(sample+"\t")
    datacard.write("\nprocess\t")
    signal_process=0
    bkg_process=1
    for sample in (sample_dict):
        for i in range(len(regions)):
            if (("signalMu" in sample) or ("signalEle" in sample)):
                datacard.write(str(signal_process)+"\t")
                signal_process-=1
            else:
                datacard.write(str(bkg_process)+"\t")
                bkg_process+=1

    datacard.write("\nrate\t")
    for sample in sample_dict:
        for i in range(len(regions)):
            datacard.write("-1\t")
    datacard.write("\n---------------\n")
    if autoMCStats:
        datacard.write("* autoMCStats 0 1 1\n")

    datacard.write("lumiRun2\tlnN\t")
    for i in range(n):
        for j in range(len(regions)):
            datacard.write("1.016\t")
    datacard.write("\n")
    for syst in syst_list:
        datacard.write(syst+"\tshape\t")
        for i in range(n):
            for j in range(len(regions)):
                datacard.write("1\t")
        datacard.write("\n")
    datacard.close()
