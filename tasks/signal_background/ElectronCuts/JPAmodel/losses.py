import torch

semileptonic_weight=(0.44 #Semileptonic BR
                    *0.33     #Muon fraction
                    )
bkg_weight=semileptonic_weight*0.5*(1-8.4e-4)
signal_weight=semileptonic_weight*0.518*8.4e-4

diLept_weight=(0.11*0.2365)

class SBLoss(torch.nn.Module):
    def __init__(self):
        super(SBLoss, self).__init__()

    def forward(self, output, target):
        s=signal_weight*torch.sum(torch.exp(output[target==1,1]))/output[target==1].shape[0]
        b=bkg_weight*torch.sum(torch.exp(output[target==0,1]))/output[target==0].shape[0]
        return -(s**2/(s+b+1e-8))
        #return (s+b)/((s+1e-8)**2)
    
    
class AsimovLoss(torch.nn.Module):
    def __init__(self):
        super(AsimovLoss, self).__init__()
        
    def forward(self, output, target):
        s=signal_weight*torch.sum(torch.exp(output[target==2,2]))/output[target==2].shape[0]
        b=bkg_weight*torch.sum(torch.exp(output[target==0,2]))/output[target==0].shape[0]
        b+=diLept_weight*torch.sum(torch.exp(output[target==1,2]))/output[target==1].shape[0]
        Z_a=torch.sqrt(2*((s+b)*torch.log(1+s/b)-s))
        #return -Z_a
        return 1/Z_a

    
class SB_Gauss_Loss(torch.nn.Module):
    def __init__(self,sigma=0.05):
        super(SB_Gauss_Loss, self).__init__()
        self.sigma=sigma


    def forward(self,output,target):
        out=torch.exp(output[:,1])
        x=torch.linspace(0,1,100).cuda()
        gaus=lambda x,mu,sigma: torch.exp(-(x-mu)**2/(2*sigma**2))/(mu.shape[1])
        gauss_bkg=bkg_weight*torch.sum(gaus(x[:,None],out[target==0][None,:],self.sigma),axis=1)
        gauss_sig=signal_weight*torch.sum(gaus(x[:,None],out[target==1][None,:],self.sigma),axis=1)
        #ratio=gauss_sig**2/(gauss_bkg)
        #return -torch.sum(ratio,axis=0)/x.shape[0]
        return -(torch.sum((x*gauss_sig/x.shape[0])**2,axis=0)/torch.sum(x*gauss_bkg/x.shape[0],axis=0))
    
    
class Brier_Score(torch.nn.Module):
    def __init__(self):
        super(Brier_Score, self).__init__()
        
    def forward(self,output,target):
        out=torch.exp(output[:,1])
        return torch.mean((out-target)**2)
    
class Score_Ratio_Loss(torch.nn.Module):
    def __init__(self):
        super(Score_Ratio_Loss, self).__init__()
    
    def forward(self,output,target):
        out=torch.exp(output[:,1])
        return -torch.sum(out[target==1]**2)/(torch.sum(out[target==0]**2)+1e-8)