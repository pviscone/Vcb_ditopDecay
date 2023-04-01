import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from Attention_block import Attention
from MLP import MLP
from torchview import draw_graph

from livelossplot import PlotLosses

from ipywidgets import Output
from IPython.display import display
import numba



# enable gpu if available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

cpu = torch.device("cpu")
device = torch.device(dev)


class JPANet(torch.nn.Module):

    def __init__(self,
                 mu_arch=None, nu_arch=None, jet_arch=None, event_arch=None, prefinal_arch=None,
                 attention_arch=None, final_arch=None, final_attention=False,
                 mu_data=None, nu_data=None, jet_data=None, label=None,
                 batch_size=1, test_size=0.15, n_heads=1,
                 weight=None,
                 jet_mean=0, jet_std=1,
                 optim={}, early_stopping=None, shuffle=False, dropout=0.15):
        super().__init__()
        assert mu_data is not None
        assert nu_data is not None
        assert jet_data is not None
        assert label is not None
        assert final_arch is not None

        self.liveloss = PlotLosses(
            groups={'Loss': ['train_loss', 'test_loss'], 'Acccuracy': ['train_accuracy','test_accuracy'], })
        self.log={}
        
        self.loss_fn = torch.nn.NLLLoss(weight=weight)
        self.early_stopping = early_stopping

        self.mu_train, self.mu_test, self.nu_train, self.nu_test, self.jet_train, self.jet_test, self.y_train, self.y_test = train_test_split(
            mu_data, nu_data, jet_data, label, test_size=test_size, shuffle=True)

        self.mu_mean = self.mu_train.mean(axis=0)
        self.mu_std = self.mu_train.std(axis=0)
        self.nu_mean = self.nu_train.mean(axis=0)
        self.nu_std = self.nu_train.std(axis=0)
        self.jet_mean = jet_mean
        self.jet_std = jet_std
        
        self.jet_test[self.jet_test==0]=-10
        self.jet_train[self.jet_train==0]=-10

        self.n_events_train = self.mu_train.shape[0]
        self.shuffle_at_each_epoch = shuffle
        self.batch_size = batch_size
        self.n_batch = np.ceil(self.mu_train.shape[0]/batch_size).astype(int)

        self.test_loss = np.array([])
        self.train_loss = np.array([])

        # Declare the layers here
        if mu_arch is not None:
            self.mu_mlp = MLP(arch=mu_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)
        else:
            self.mu_mlp=torch.nn.Identity()

        
        if nu_arch is not None:
            self.nu_mlp = MLP(arch=nu_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)

        else:
            self.nu_mlp=torch.nn.Identity()


        self.mu_nu_norm = torch.nn.LayerNorm(event_arch[0])
        
        if event_arch is not None:
            self.ev_mlp = MLP(arch=event_arch,out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)
        else:
            self.ev_mlp = torch.nn.Identity()

        if jet_arch is not None:
            self.mlp_jet = MLP(arch=jet_arch,out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)
            attention_input_dim = jet_arch[-1]
        else:
            self.mlp_jet = torch.nn.Identity()
            attention_input_dim = self.jet_train.shape[2]

        
        if attention_arch is not None:
            self.attention = Attention(input_dim=attention_input_dim, mlp_arch=attention_arch, n_heads=n_heads, dropout=dropout)
        else:
            self.attention = torch.nn.Identity()
            self.attention.forward = lambda x,key_padding_mask=None: x


        if prefinal_arch is not None:
            self.prefinal_mlp = MLP(arch=prefinal_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)
        else:
            self.prefinal_mlp = torch.nn.Identity()

        if final_attention:
            self.total_norm=torch.nn.Identity()
            self.total = Attention(input_dim=final_arch[0], mlp_arch=final_arch, n_heads=n_heads, dropout=dropout)
        else:
            self.total_norm = torch.nn.LayerNorm(final_arch[0])
            self.total = MLP(arch=final_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)
            self.total.forward=lambda x,key_padding_mask=None:self.total.forward(x)
    
        self.output=MLP(arch=[final_arch[-1],self.y_test.shape[-1]],out_activation=torch.nn.LogSoftmax(dim=1), dropout=None)

        self.optim_dict = optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)

        self.train_accuracy = []
        self.test_accuracy = []

        self.permutation_matrix=torch.tensor(perm_matrix(self.jet_test.shape[1]),device=device,dtype=torch.float32)

    def forward(self, mu, nu, jet):

        pad_mask = ((jet == -10)[:, :, 0].squeeze()).to(torch.bool)
        
        mu = (mu-self.mu_mean)/(self.mu_std+1e-5)
        nu = (nu-self.nu_mean)/(self.nu_std+1e-5)
        jet = (jet-self.jet_mean)/(self.jet_std+1e-5)

        out_mu = self.mu_mlp(mu)
        out_nu = self.nu_mlp(nu)
        out_ev = self.ev_mlp(torch.cat((out_mu, out_nu), dim=2))

        del out_mu, out_nu

        out_jet = self.mlp_jet(jet)
        out_jet = self.attention(out_jet,key_padding_mask=pad_mask)
        
        total_out = torch.cat((out_ev, out_jet), dim=1)
        total_out = self.total_norm(total_out)
        total_out = self.prefinal_mlp(total_out)
        
        pad_mask = torch.cat((torch.tensor(np.zeros(
            out_ev.shape[0]), device=device, dtype=torch.bool).unsqueeze(1), pad_mask), dim=1)
        total_out = self.total(total_out,query=out_jet,key_padding_mask=pad_mask)
        total_out = self.output(total_out)

        return total_out

    def train_loop(self, epochs,show_each=False):
        epoch_loop = tqdm(range(epochs), desc="epoch")

        out = Output()
        display(out)
        for epoch in epoch_loop:
            self.train()
            mu_train = self.mu_train
            nu_train = self.nu_train
            jet_train = self.jet_train
            y_train = self.y_train
            if self.shuffle_at_each_epoch:
                perm = torch.randperm(self.mu_train.size()[0])
                mu_train = mu_train[perm]
                nu_train = nu_train[perm]
                jet_train = jet_train[perm]
                y_train = y_train[perm]

            # for x_batch, y_batch in self.data_loader:
            for i in range(self.n_batch):
                if ((i+1)*self.batch_size < self.n_events_train):
                    mu_batch = mu_train[i *
                                        self.batch_size:(i+1)*self.batch_size]
                    nu_batch = nu_train[i *
                                        self.batch_size:(i+1)*self.batch_size]
                    jet_batch = jet_train[i *
                                          self.batch_size:(i+1)*self.batch_size]
                    y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                else:
                    mu_batch = mu_train[i*self.batch_size:self.n_events_train]
                    nu_batch = nu_train[i*self.batch_size:self.n_events_train]
                    jet_batch = jet_train[i *
                                          self.batch_size:self.n_events_train]
                    y_batch = y_train[i*self.batch_size:self.n_events_train]

                y_logits = self.forward(
                    mu_batch, nu_batch, jet_batch)
                
                train_loss_step = self.loss_fn(
                    y_logits, y_batch)
                self.optimizer.zero_grad()
                train_loss_step.backward()
                self.optimizer.step()

            self.eval()
            with torch.inference_mode():

                test_logits = self.forward(
                    self.mu_test, self.nu_test, self.jet_test)

                test_loss_step = self.loss_fn(
                    test_logits, self.y_test)
                #print(f"Test loss: {test_loss_step.to(cpu).numpy()}, Train loss: {train_loss_step.to(cpu).numpy()}")
                self.test_loss = np.append(
                    self.test_loss, test_loss_step.to(cpu).numpy())

                self.train_loss = np.append(
                    self.train_loss, train_loss_step.to(cpu).numpy())

                classified_test=self.classify(test_logits)
                classified_test=torch.eq(classified_test,self.y_test).cpu().numpy()
                classified_test=np.bitwise_and.reduce(classified_test,axis=1)

                self.test_accuracy.append(
                    classified_test.sum()/len(classified_test))


                idx=np.random.choice(range(self.y_train.shape[0]),self.y_test.shape[0],replace=False)
                
                classified_train=self.classify(self(self.mu_train[idx],self.nu_train[idx],self.jet_train[idx]))
                classified_train=torch.eq(classified_train,self.y_train[idx]).cpu().numpy()
                classified_train=np.bitwise_and.reduce(classified_train,axis=1)
                
                self.train_accuracy.append(classified_train.sum()/len(classified_train))
                    
                    
                    
                if(show_each):

                    self.log["train_loss"]=self.train_loss[-1]
                    self.log["test_loss"]=self.test_loss[-1]
                    self.log["test_accuracy"]=self.test_accuracy[-1]
                    self.log["train_accuracy"]=self.train_accuracy[-1]
                    with out:
                        self.liveloss.update(self.log)
                        if(epoch%show_each==0):
                            self.liveloss.send()
                            self.liveloss.draw()
                    
                    
                #classified_train=torch.argmax(self(self.x_train),dim=1) == self.y_train.squeeze()

                # self.train_accuracy.append(classified_train.sum().item()/len(classified_train))

            if self.early_stopping != None:
                if not "Patience" in self.early_stopping:
                    raise ValueError(
                        "early_stopping must be a dictionary with key 'Patience'")
                patience = self.early_stopping["Patience"]
                if epoch > patience:
                    if "RMS" in self.early_stopping:
                        if np.std(self.test_loss[-patience:]) < self.early_stopping["RMS"]:
                            print("Early stopping: test loss not changing")
                            break
                    if "Mean" in self.early_stopping and self.early_stopping == True:
                        if np.mean(self.test_loss[-patience:]) > np.mean(self.test_loss[-2*patience:-patience]):
                            print("Early stopping: test loss increasing")
                            break



    def loss_plot(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, label="test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracy, label="train")
        plt.plot(self.test_accuracy, label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

    def graph(self):
        model_graph = draw_graph(self, input_data=[
                                 self.mu_test[:self.batch_size], self.nu_test[:self.batch_size], self.jet_test[:self.batch_size]], expand_nested=True,depth=5)
        return model_graph.visual_graph

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def classify(self,out):
        perm_sum_arr=torch.einsum("bli,ail->ba",out,self.permutation_matrix)
        best_perm_arr=torch.argmax(perm_sum_arr,axis=1)
        best_perm_masks=torch.swapaxes(self.permutation_matrix[best_perm_arr],1,2)
        return torch.argmax(best_perm_masks,axis=1)
    
    

@numba.jit(nopython=True,fastmath=True)
def perm_matrix(n_jet):
    res=np.zeros((1,4,n_jet))
    prod=np.zeros((1,4,n_jet))
    for i1 in range(n_jet):
        for i2 in range(n_jet):
            for i3 in range(n_jet):
                for i4 in range(n_jet):
                    i=np.array([i1,i2,i3,i4])
                    arr_check=np.unique(i)
                    if(len(arr_check)==4):
                        for j in range(4):
                            prod[0,j,i[j]]=(1.)
                        res=np.concatenate((res,prod),axis=0)
                        prod=np.zeros((1,4,n_jet))
    return res[1:]