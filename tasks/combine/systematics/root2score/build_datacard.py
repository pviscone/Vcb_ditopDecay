
def build_datacard(sample_dict,syst_dict,autoMCStats=True):
    n=len(sample_dict)
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
        datacard.write("Muons\t")
        datacard.write("Electrons\t")
    datacard.write("\nprocess\t")
    for sample in sample_dict:
        datacard.write(sample+"\t")
        datacard.write(sample+"\t")
    datacard.write("\nprocess\t")
    signal_process=0
    bkg_process=1
    for sample in (sample_dict):
        if (("signalMu" in sample) or ("signalEle" in sample)):
            datacard.write(str(signal_process)+"\t")
            signal_process-=1
            datacard.write(str(signal_process)+"\t")
            signal_process-=1
        else:
            datacard.write(str(bkg_process)+"\t")
            bkg_process+=1
            datacard.write(str(bkg_process)+"\t")
            bkg_process+=1
    datacard.write("\nrate\t")
    for sample in sample_dict:
        datacard.write("-1\t")
        datacard.write("-1\t")
    datacard.write("\n---------------\n")
    if autoMCStats:
        datacard.write("* autoMCStats 0 1 1\n")

    datacard.write("lumiRun2\tlnN\t")
    for i in range(n):
        datacard.write("1.016\t")
        datacard.write("1.016\t")
    datacard.write("\n")
    for syst in syst_dict:
        datacard.write(syst+"\tshape\t")
        for i in range(n):
            datacard.write("1\t")
            datacard.write("1\t")
        datacard.write("\n")
    datacard.close()
