#%%
import json
import copy
import argparse




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="datacard json", type=str, required=True)
    parser.add_argument("-o", "--output", help="output datacard", type=str, required=True)
    parser.add_argument("-dG", "--disable_group", help="syst group to disable", nargs="+", required=False,default=[])
    parser.add_argument("-dC", "--disable_channel", help="channel to disable", nargs="+", required=False,default=[])
    args = parser.parse_args()
    
    json_file = json.load(open(args.input))
    output = open(args.output, "w")
    
    output.write(f'''
imax *
jmax *
kmax *
---------------
shapes * * {json_file["hist_name"]} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_syst_$SYSTEMATIC
---------------
bin    ''')
    
    
    channels_dict=json_file["channels"]
    channels_dict_temp=copy.copy(channels_dict)
    for channel in channels_dict_temp:
        if channel in args.disable_channel:
            del channels_dict[channel]
    for channel in channels_dict:
        channels_dict[channel]["signal"]=channels_dict[channel]["signal"]+json_file["commonSig"]
        channels_dict[channel]["bkg"]=channels_dict[channel]["bkg"]+json_file["commonBkg"]
        output.write(f"{channel}    "*len(channels_dict[channel]["signal"]))
        output.write(f"{channel}    "*len(channels_dict[channel]["bkg"]))
    output.write("\nprocess    ")
    for channel in channels_dict:
        for proc in channels_dict[channel]["signal"]:
            output.write(f"{proc}    ")
        for proc in channels_dict[channel]["bkg"]:
            output.write(f"{proc}    ")
    output.write("\nprocess    ")
    
    i_sig=0
    i_bkg=1
    for channel in channels_dict:
        for proc in channels_dict[channel]["signal"]:
            output.write(f"{i_sig}    ")
            i_sig-=1
        for proc in channels_dict[channel]["bkg"]:
            output.write(f"{i_bkg}    ")
            i_bkg+=1
    output.write("\nrate    ")
    for channel in channels_dict:
        for proc in channels_dict[channel]["signal"]:
            output.write(f"-1    ")
        for proc in channels_dict[channel]["bkg"]:
            output.write(f"-1    ")
    output.write("\n---------------\n")
    
    systs_groups=json_file["systs"]
    del systs_groups["notuse"]
    systs_groups_temp=copy.copy(systs_groups)
    for group in systs_groups_temp:
        if group in args.disable_group:
            del systs_groups[group]
    
    
    for group in systs_groups:
        for syst in systs_groups[group]:
            output.write(f"{syst}    {systs_groups[group][syst]['type']}   ")
            region=systs_groups[group][syst]["region"]
            process=systs_groups[group][syst]["proc"]    
            for channel in channels_dict:
                for proc in channels_dict[channel]["signal"]:  
                    if ((channel in region or region=='all') and 
                        (proc in process or process=='all')):
                        
                        output.write(f"{systs_groups[group][syst]['value']}    ")
                    else:
                        output.write(f"-    ")
                for proc in channels_dict[channel]["bkg"]:
                    if ((channel in region or region=='all') and 
                        (proc in process or process=='all')):
                        
                        output.write(f"{systs_groups[group][syst]['value']}    ")
                    else:
                        output.write(f"-    ")
            output.write("\n")
            
    for group in systs_groups:
        output.write(f"{group}    group    =    ")
        for syst in systs_groups[group]:
            output.write(f"{syst}    ")
        output.write("\n")
            

    