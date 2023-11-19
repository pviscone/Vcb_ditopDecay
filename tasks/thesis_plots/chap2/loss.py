#%%
import torch
import matplotlib.pyplot as plt
import mplhep


muon_loss=torch.load('/scratchnvme/pviscone/Vcb_ditopDecay/tasks/signal_background/MuonCuts/loss_final.pt')

electron_loss=torch.load('/scratchnvme/pviscone/Vcb_ditopDecay/tasks/signal_background/ElectronCuts/loss.pt')
# %%
mplhep.style.use("CMS")
def plot_loss(loss, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(loss["train_loss"], label="Train",linewidth=2)
    ax.plot(loss["test_loss"], label="Validation",linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    mplhep.cms.lumitext(title+" channel", ax=ax)
    #ax.text(0.425, 0.95, title, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    #ax.set_yscale('log')
    ax.legend()
    ax.grid()
    mplhep.cms.text("Private Work", ax=ax)
    plt.savefig(f'loss_{title}.pdf', bbox_inches='tight')
    plt.show()
    
    return fig, ax

plot_loss(muon_loss, 'Muon')

plot_loss(electron_loss, 'Electron')