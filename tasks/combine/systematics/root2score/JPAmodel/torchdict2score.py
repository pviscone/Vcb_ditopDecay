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
