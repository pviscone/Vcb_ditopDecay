import torch
from root2score.JPAmodel.JPANet import JPANet

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
cpu = torch.device("cpu")
device = torch.device(dev)

def create_model(weight_path,device=device):
    mu_feat=3
    nu_feat=3
    jet_feat=6
    model = JPANet(mu_arch=None, nu_arch=None, jet_arch=[jet_feat, 128, 128],
               jet_attention_arch=[128,128,128],
               event_arch=[mu_feat+nu_feat, 128, 128,128],
               masses_arch=[36,128,128],
               pre_attention_arch=None,
               final_attention=True,
               post_attention_arch=[128,128],
               secondLept_arch=[3,128,128],
               post_pooling_arch=[128,128,64],
               n_heads=2, dropout=0.02,
               early_stopping=None,
               n_jet=7,
               device=device,
               )

    state_dict=torch.load(weight_path,map_location=torch.device(device))
    state_dict.pop("loss_fn.weight")
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def predict(model,dataset,bunch=1):
    model.eval()
    with torch.inference_mode():
        score=torch.exp(model.predict(dataset,bunch=bunch)[:,-1]).detach().to(cpu).numpy().astype("float")
    return score


def torchdict2score(torch_dict,bunch=1,device=device):
    model={"Muons":create_model("root2score/JPAmodel/state_dict_Muons.pt",device=device),
            "Electrons":create_model("root2score/JPAmodel/state_dict_Electrons.pt",device=device)}
    score_dict={}
    for cut in torch_dict:
        score_dict[cut]={}
        for dataset in torch_dict[cut]:
            score_dict[cut][dataset]={}
            for syst in torch_dict[cut][dataset]:
                print(f"{cut}: {dataset}_{syst}")
                score_dict[cut][dataset][syst]=predict(model[cut],torch_dict[cut][dataset][syst],bunch=bunch)
                
    return score_dict
