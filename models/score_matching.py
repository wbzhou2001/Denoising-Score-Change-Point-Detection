import torch
import torch.nn as nn
import numpy as np
import os

class Dense(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Dense, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.relu  = nn.ReLU()

    def forward(self, x):
        activation = self.linear(x)
        activation = self.relu(activation)
        return activation
    
class ScoreMatching(nn.Module):

    def __init__(self,
                 data_dim, hidden_layers,
                 device = 'cpu'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # init score network
        nn_list = [ Dense(data_dim, hidden_layers[0]) ]
        for i in range(len(hidden_layers) - 1):
            nn_list = nn_list + [ Dense(hidden_layers[i], hidden_layers[i + 1]) ]
        nn_list = nn_list + [ nn.Linear(hidden_layers[-1], data_dim) ]
        self.fc = nn.Sequential(*nn_list).to(self.device)

    def fit(self, data,
            num_epochs, lr, batch_size, save_path,
            patience = 50):
        '''
        Args:
        - [ batch_size, data_dim ] th
        '''
        os.makedirs(save_path, exist_ok=True)
        D_tr = data.clone().requires_grad_(True).float()
        dataset     = torch.utils.data.TensorDataset(D_tr)
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 
        optim       = torch.optim.Adam(self.parameters(), lr=lr)
        losses = [] 
        for i in range(num_epochs):
            losses_epoch = []
            for batch in dataloader:
                x = batch[0].to(self.device)    # [ batch_size, data_dim ] th
                optim.zero_grad()
                loss = self(x).mean()           # scalar th
                loss.backward()
                optim.step()
                losses_epoch.append(loss.item())
            losses.append(np.mean(losses_epoch))
            
            # log
            if i % (num_epochs // 10) == 0:
                print(f'Epoch = {i} \t Loss = {losses[-1]}')
                torch.save(self.state_dict(), save_path + f'/iter_{i // (num_epochs // 10) + 1}.pth')

            # early stop
            min_id      = np.where(losses == np.min(losses))[0].item()
            if i - min_id >= patience:
                print(f'Training has not improved in the last {patience} epochs! Early stopped!')
                break
        torch.save(self.state_dict(), save_path + f'/state_dict.pth')
        
    def load(self, save_path):
        state_dict = torch.load(save_path + '/state_dict.pth')
        self.load_state_dict(state_dict)
        print('Found saved model and loaded!')
    
    def forward(self, x):
        '''
        Hyvarinen score
        Args:
        - x:    [ batch_size, data_dim ] th
        Returns:
        - val:  [ batch_size ] th, the hyvarinen score
        '''
        s   = self.score(x)
        div = self.div(s, x)
        val = 1 / 2 * torch.norm(s, dim=1)**2 + div
        return val

    def score(self, x):
        '''
        Score function
        Args:
        - x:    [ batch_size, data_dim ] th
        Returns:
        - s:    [ batch_size, data_dim ] th
        '''
        s = self.fc(x)
        return s

    def div(self, s, x):
        '''
        Divergence of score function
        Args:
        - s:    [ batch_size, data_dim ] th
        - x:    [ batch_size, data_dim ] th
        Returns:
        - div:  [ batch_size ] th
        '''
        div = []
        for i in range(s.shape[-1]):
            grad = torch.autograd.grad(
                outputs = s[:, i].sum(),
                inputs  = x,
                retain_graph = True,
                create_graph= True
            )[0][:, i]                          # [ batch_size ] th
            div.append(grad)
        div = torch.stack(div, -1).sum(-1)      # [ batch_size ] th
        return div

    def H_score(self, x):
        '''
        Get Hyvarinen score
        Args:
        - x:    [ batch_size, data_dim ] np
        Returns:
        - val:  [ batch_size ] np, the hyvarinen score
        '''
        self.eval()
        x_ = torch.tensor(x).float().requires_grad_(True).to(self.device)
        s   = self.score(x_)
        div = self.div(s, x_)
        val = 1 / 2 * torch.norm(s, dim=1)**2 + div
        return val.detach().cpu().numpy()

class DenoisingScoreMatching(nn.Module):

    def __init__(self,
                 data_dim, hidden_layers, sigmas,
                 device = 'cpu'):
        '''
        Args:
        - sigmas: [ nsteps ] th --- noise scales, equivalent to a variance exploding (VE) scheduler in a diffusion process
        '''
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sigmas = sigmas
        self.nsteps = len(sigmas)

        # init score network
        fcs = []
        for i in range(self.nsteps):
            nn_list = [ Dense(data_dim, hidden_layers[0]) ]
            for i in range(len(hidden_layers) - 1):
                nn_list = nn_list + [ Dense(hidden_layers[i], hidden_layers[i + 1]) ]
            nn_list = nn_list + [ nn.Linear(hidden_layers[-1], data_dim) ]
            fc = nn.Sequential(*nn_list).to(self.device)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)

    def fit(self, data,
            num_epochs, lr, batch_size, save_path,
            patience = 50):
        '''
        Args:
        - data: [ batch_size, data_dim ] th or np
        '''
        os.makedirs(save_path, exist_ok=True)
        if isinstance(data, np.ndarray):
            data = torch.tensor(data).float()
        D_tr        = data.requires_grad_(True).float()
        dataset     = torch.utils.data.TensorDataset(D_tr)
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 
        optim       = torch.optim.Adam(self.parameters(), lr=lr)
        losses = [] 
        for i in range(num_epochs):
            losses_epoch = []
            for batch in dataloader:
                x = batch[0].to(self.device)    # [ batch_size, data_dim ] th
                optim.zero_grad()
                loss = self.loss(x)
                loss.backward()
                optim.step()
                losses_epoch.append(loss.item())
            losses.append(np.mean(losses_epoch))
            
            # log
            if i % (num_epochs // 10) == 0:
                print(f'Epoch = {i} \t Loss = {losses[-1]}')
                torch.save(self.state_dict(), save_path + f'/iter_{i // (num_epochs // 10) + 1}.pth')

            # early stop
            min_id      = np.where(losses == np.min(losses))[0].item()
            if i - min_id >= patience:
                print(f'Training has not improved in the last {patience} epochs! Early stopped!')
                break
        torch.save(self.state_dict(), save_path + f'/state_dict.pth')

    def loss(self, data):
        '''
        Args:
        - data:       [ batch_size, data_dim ] th,
        Returns:
        - loss:       [ batch_size ] th 
        '''
        loss = 0.
        for i in range(self.nsteps):
            noise       = torch.randn_like(data) * self.sigmas[i]   # [ batch_size, data_dim ] th
            noisy_data  = data + noise
            true_score  = (data - noisy_data) / self.sigmas[i]**2   # [ batch_size, data_dim ] th
            est_score   = self.score(noisy_data, i, otype='torch')                    # [ batch_size, data_dim ] th
            loss_step   = 1/2 * torch.mean((est_score - true_score)**2)
            loss += self.sigmas[i]**2 * loss_step
        return loss / self.nsteps
        
    def load(self, save_path):
        state_dict = torch.load(save_path + '/state_dict.pth')
        self.load_state_dict(state_dict)
        print('Found saved model and loaded!')

    def score(self, x, i = 0, otype = 'numpy'):
        '''
        Stein score function
        Args:
        - x:    [ batch_size, data_dim ] th
        - i:    which score component id in self.fcs to use
        Returns:
        - s:    [ batch_size, data_dim ] th or np
        '''
        if otype == 'torch':
            s = self.fcs[i](x)
        elif otype == 'numpy':
            self.eval()
            with torch.no_grad():
                s = self.fcs[i](x)
                s = s.cpu().numpy()
        else:
            raise NotImplementedError('Unrecognized otype!')
        return s
    
    def H_score(self, x, i = 0):
        '''
        Args:
        - x:        [ batch_size, data_dim ] th or np --- input to the score function, requires_grad = True 
        Returns:
        - H:        [ batch_size ] np --- Hyvarinen score function
        '''
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float().requires_grad_(True).to(self.device)
        score = self.score(x, i, otype = 'torch')       # [ batch_size, data_dim ] th --- Stein score function
        div = []
        for i in range(score.shape[-1]):
            grad = torch.autograd.grad(
                outputs = score[:, i].sum(),
                inputs  = x,
                retain_graph = True,
                create_graph= True
            )[0][:, i]                                  # [ batch_size ] th
            div.append(grad)
        div = torch.stack(div, -1).sum(-1)              # [ batch_size ] th
        H = 1 / 2 * torch.norm(score, dim=1)**2 + div   # [ batch_size ] th
        return H.cpu().detach().numpy()