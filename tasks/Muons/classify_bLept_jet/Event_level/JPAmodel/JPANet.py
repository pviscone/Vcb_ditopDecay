import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from Attention_block import Attention
from MLP import MLP
from torchview import draw_graph
#from torch.utils.data.dataloader import default_collate


# enable gpu if available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

cpu = torch.device("cpu")
device = torch.device(dev)


class JPANet(torch.nn.Module):

    def __init__(self,
                 mu_data=None, nu_data=None, jet_data=None, label=None,
                 mlp_arch=None, batch_size=1, test_size=0.15, n_heads=1,
                 optim={}, early_stopping=None, shuffle=False, dropout=0.15):
        super().__init__()
        assert mu_data is not None
        assert nu_data is not None
        assert jet_data is not None
        assert label is not None
        assert mlp_arch is not None

        self.loss_fn = torch.nn.NLLLoss()
        self.early_stopping = early_stopping

        self.mu_train, self.mu_test, self.nu_train, self.nu_test, self.jet_train, self.jet_test, self.y_train, self.y_test = train_test_split(
            mu_data, nu_data, jet_data, label, test_size=test_size, shuffle=True)

        self.mu_mean = self.mu_train.mean(axis=0)
        self.mu_std = self.mu_train.std(axis=0)
        self.nu_mean = self.nu_train.mean(axis=0)
        self.nu_std = self.nu_train.std(axis=0)
        self.jet_mean = self.jet_train.mean(axis=0)
        self.jet_std = self.jet_train.std(axis=0)

        self.n_events_train = self.mu_train.shape[0]

        self.shuffle_at_each_epoch = shuffle
        self.batch_size = batch_size
        self.n_batch = np.ceil(self.mu_train.shape[0]/batch_size).astype(int)

        self.test_loss = np.array([])
        self.train_loss = np.array([])

        # Declare the layers here
        mu_arch = [self.mu_train.shape[2]]+mlp_arch
        self.mu_mlp = MLP(
            arch=mu_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)

        nu_arch = [self.nu_train.shape[2]]+mlp_arch
        self.nu_mlp = MLP(
            arch=nu_arch, out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)

        self.mu_nu_norm = torch.nn.LayerNorm(mlp_arch[-1])
        self.ev_mlp = MLP(arch=[mlp_arch[-1]*2]+mlp_arch,
                          out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)

        self.mlp_jet = MLP(arch=[self.jet_train.shape[2]]+mlp_arch,
                           out_activation=torch.nn.LeakyReLU(0.1), dropout=dropout)

        self.attention = Attention(
            input_dim=mlp_arch[-1], mlp_arch=mlp_arch, n_heads=n_heads, dropout=dropout)

        self.total_norm = torch.nn.LayerNorm(mlp_arch[-1]*2)
        self.total_mlp = MLP(arch=[mlp_arch[-1]*2]+mlp_arch+[1],
                             out_activation=torch.nn.LogSoftmax(dim=1), dropout=dropout)

        self.optim_dict = optim
        self.optimizer = torch.optim.Adam(self.parameters(), **self.optim_dict)

        self.train_accuracy = []
        self.test_accuracy = []

    def forward(self, mu, nu, jet):
        mu = (mu-self.mu_mean)/(self.mu_std+1e-5)
        nu = (nu-self.nu_mean)/(self.nu_std+1e-5)
        jet = (jet-self.jet_mean)/(self.jet_std+1e-5)

        out_mu = self.mu_mlp(mu)
        out_nu = self.nu_mlp(nu)
        out_ev = self.ev_mlp(torch.cat((out_mu, out_nu), dim=2))

        del out_mu, out_nu

        out_jet = self.mlp_jet(jet)
        out_jet = self.attention(out_jet)
        out_ev = out_ev.repeat((1, jet.shape[1], 1))
        total_out = self.total_norm(torch.cat((out_ev, out_jet), dim=2))
        total_out = self.total_mlp(total_out)
        return total_out

    def train_loop(self, epochs):
        epoch_loop = tqdm(range(epochs), desc="epoch")

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
                    mu_batch, nu_batch, jet_batch).squeeze()
                train_loss_step = self.loss_fn(
                    y_logits.squeeze(), y_batch.squeeze())
                self.optimizer.zero_grad()
                train_loss_step.backward()
                self.optimizer.step()

            self.eval()
            with torch.inference_mode():

                test_logits = self.forward(
                    self.mu_test, self.nu_test, self.jet_test).squeeze()

                test_loss_step = self.loss_fn(
                    test_logits, self.y_test.squeeze())
                print(
                    f"Test loss: {test_loss_step.to(cpu).numpy()}, Train loss: {train_loss_step.to(cpu).numpy()}")
                self.test_loss = np.append(
                    self.test_loss, test_loss_step.to(cpu).numpy())

                self.train_loss = np.append(
                    self.train_loss, train_loss_step.to(cpu).numpy())

                classified_test = torch.argmax(
                    test_logits, dim=1) == self.y_test.squeeze()

                self.test_accuracy.append(
                    classified_test.sum().item()/len(classified_test))

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
                                 self.mu_test[:self.batch_size], self.nu_test[:self.batch_size], self.jet_test[:self.batch_size]], expand_nested=True)
        return model_graph.visual_graph

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
