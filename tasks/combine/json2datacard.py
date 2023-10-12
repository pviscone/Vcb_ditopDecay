import json
import argparse




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="datacard json", type=str, required=True)
    parser.add_argument("-o", "--output", help="output datacard", type=str, required=True)
    parser.add_argument("-dG", "--disable_group", help="syst group to disable", nargs="+", required=False,default=[])
    parser.add_argument("-dC", "--disable_channell", help="channell to disable", nargs="+", required=False,default=[])
    args = parser.parse_args()
    
    json_file = json.load(open(args.input))
    output = open(args.output, "w")
    
    for chan in args.disable_channell:
        json_file["channels"].pop(chan)
    
    channels_dict=json_file["channels"]
    
    n_channels=len(channels_dict)
    n_tot=n_channels*(len(json_file["commonSig"])+len(json_file["commonBkg"]))+sum([len(channels_dict[channel]["signal"])+len(channels_dict[channel]["bkg"]) for channel in channels_dict])
    
    
    output.write("imax *\n")
    output.write("jmax *\n")
    output.write("kmax *\n")
    output.write("---------------\n")
    output.write(f'shapes * * {json_file["hist_name"]} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_syst_$SYSTEMATIC\n')
    output.write("---------------\n")
    output.write("bin\t")
    
    proc_order=[]
    for channel in channels_dict:
        n_proc=len(channels_dict[channel]["signal"])+len(json_file["commonSig"])+len(channels_dict[channel]["bkg"])+len(json_file["commonBkg"])
        for i in range(n_proc):
            output.write(f"{channel}\t")
        
    output.write("\nprocess\t")
    
    for channel in channels_dict:
        for proc in channels_dict[channel]["signal"]:
            output.write(f"{proc}\t")
            proc_order.append(proc)
        for proc in json_file["commonSig"]:
            output.write(f"{proc}\t")
            proc_order.append(proc)
        for proc in channels_dict[channel]["bkg"]:
            output.write(f"{proc}\t")
            proc_order.append(proc)
        for proc in json_file["commonBkg"]:
            output.write(f"{proc}\t")
            proc_order.append(proc)
            
    output.write("\nprocess\t")
    
    i_sig=0
    i_bkg=1
    for channel in channels_dict:
        for proc in channels_dict[channel]["signal"]:
            output.write(f"{i_sig}\t")
            i_sig-=1
        for proc in json_file["commonSig"]:
            output.write(f"{i_sig}\t")
            i_sig-=1
        for proc in channels_dict[channel]["bkg"]:
            output.write(f"{i_bkg}\t")
            i_bkg+=1
        for proc in json_file["commonBkg"]:
            output.write(f"{i_bkg}\t")
            i_bkg+=1
    output.write("\nrate\t")
    
    for channel in channels_dict:
        n_proc=len(channels_dict[channel]["signal"])+len(json_file["commonSig"])+len(channels_dict[channel]["bkg"])+len(json_file["commonBkg"])
        for i in range(n_proc):
            output.write(f"-1\t")
            
    output.write("\n---------------\n")
    if json_file["autoMCStats"]:
        output.write("* autoMCStats 0\t1\t1\n")
    
    
    systs_groups=(json_file["systs"])
    systs_groups.pop("notuse")
    for group in systs_groups:
        if group in args.disable_group:
            continue
        
        for syst in systs_groups[group]:
            syst_type=systs_groups[group][syst]["type"]
            output.write(f"{syst}\t{syst_type}\t")
            if systs_groups[group][syst]["proc"]=="all":
                for i in range(n_tot):
                    output.write(f'{systs_groups[group][syst]["value"]}\t')
            else:
                procs=systs_groups[group][syst]["proc"]
                for proc in procs:
                    idxs=[i for i in range(len(proc_order)) if proc_order[i] == proc]
                    for i in range(n_tot):
                        if i in idxs:
                            output.write(f'{systs_groups[group][syst]["value"]}\t')
                        else:
                            output.write(f'-\t')
            output.write("\n")
        
    for group in systs_groups:
        output.write(f"{group}\tgroup\t=\t")
        for syst in systs_groups[group]:
            output.write(f"{syst}\t")
        output.write("\n")
    output.close()
        

                
                
