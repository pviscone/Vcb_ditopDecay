#%%
import torch

main=torch.load("all_but_others/score_dict.pt")
others=torch.load("others/score_dict_others.pt")


#%%

for channel in others["score_dict"].keys():
    for process in others["score_dict"][channel].keys():
        main["score_dict"][channel][process]=others["score_dict"][channel][process]
        main["weight_dict"][channel][process]=others["weight_dict"][channel][process]
# %%
torch.save(main,"all_score_dict.pt")