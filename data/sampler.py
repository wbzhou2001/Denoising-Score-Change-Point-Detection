import numpy as np
import torch
from torch.distributions import MultivariateNormal

class GaussianMixtureSampler:
    '''
    A sampler for synthetic two sample data using a series of Gaussain Mixture
    '''
    def __init__(self, mu0 = np.array([[-5.]]), mu1 = np.array([[5.]]),
                 scale0 = 1., scale1 = 1., noise_scale = 0.):
        '''
        Args:
        - mu:           [ ncomp, data_dim ] np, the mean of GMs
        - scale:        [ ncomp ] np, the scale of varaince of GMs 
        - noise_scale:  scaler --- optinonal, the pertubration Gaussian noise scale
        '''
        self.ncomp0, self.data_dim = mu0.shape
        self.ncomp1, self.data_dim = mu1.shape
        mu0, mu1 = torch.tensor(mu0).float(), torch.tensor(mu1).float()
        cov0 = torch.eye(self.data_dim).unsqueeze(-1).repeat(1, 1, self.ncomp0) * (torch.tensor(scale0).float() + noise_scale)   # [ data_dim, data_dim, ncomp ] th
        cov1 = torch.eye(self.data_dim).unsqueeze(-1).repeat(1, 1, self.ncomp1) * (torch.tensor(scale1).float() + noise_scale)   # [ data_dim, data_dim, ncomp ] th
        cov0, cov1 = cov0.permute(2, 0, 1), cov1.permute(2, 0, 1)   # [ ncomp, data_dim, data_dim ] th
        self.p0 = [ MultivariateNormal(mu, sigma) for mu, sigma in zip(mu0, cov0) ]
        self.p1 = [ MultivariateNormal(mu, sigma) for mu, sigma in zip(mu1, cov1) ]

    def sample(self, batch_size, n0, n1):
        '''
        Returns:
        - data: [ batch_size, n0 + n1, data_dim ]
        '''
        with torch.no_grad():
            u0 = np.random.choice(self.ncomp0, n0)
            u1 = np.random.choice(self.ncomp1, n1)
            
            # first dataset
            D0 = []
            for i in range(len(self.p0)):
                nsample = (i == u0).sum().item()    # scalar
                sample = self.p0[i].sample([batch_size, nsample,]) # [ batch_size, nsample, data_dim ] th
                D0.append(sample.numpy())
            D0 = np.concatenate(D0, 1)          # [ batch_size, n0, data_dim ] np
            indices = np.random.permutation(n0) # [ n0 ] np 
            D0 = D0[:, indices, :]              # [ batch_size, nu, data_dim ] np
            
            # second dataset
            D1 = []
            for i in range(len(self.p1)):
                nsample = (i == u1).sum().item()    # scalar
                sample = self.p1[i].sample([batch_size, nsample,]) # [ batch_size, nsample, data_dim ] th
                D1.append(sample.numpy())
            D1 = np.concatenate(D1, 1)          # [ batch_size, n1, data_dim ] np
            indices = np.random.permutation(n1) # [ n1 ] np 
            D1 = D1[:, indices, :]              # [ batch_size, n1, data_dim ] np

            D = np.concatenate([D0, D1], 1)     # [ nsample, data_dim ] np
            return D                            # [ batch_size, n0 + n1, data_dim ] np

    def pdf(self, xgrids, dtype = 0, otype = 'numpy'):
        '''
        Args:
        - xgrids:   [ batch_size, data_dim ] np
        - dtype:    pre-change = 0, post-change = 1 --- desired dataset type 
        = otype:    torch or numpy --- desired return type
        Returns:
        - [ batch_size ] th or np
        '''
        if isinstance(xgrids, np.ndarray):
            xgrids = torch.tensor(xgrids).float()
        
        pdf_comps = []
        if dtype == 0: 
            for p0 in self.p0:
                pdf_comps.append(p0.log_prob(xgrids).exp()) # append [ batch_size ]
        elif dtype == 1:
            for p1 in self.p1:
                pdf_comps.append(p1.log_prob(xgrids).exp()) # append [ batch_size ]
        else:
            raise NotImplementedError(f'Got unexpected type: {dtype}.')

        pdf = torch.stack(pdf_comps, -1)    # [ batch_size, ncomps ] th
        pdf = pdf.mean(-1)                   # [ batch_size ] th

        if otype == 'numpy':
            pdf = pdf.detach().numpy()      # [ batch_size ] np
        elif otype == 'torch':
            pass
        else:
            raise NotImplementedError('Unrecognized otype!')

        return pdf
    
    def score(self, xgrids, dtype = 0, otype = 'numpy'):
        '''
        Args:
        - xgrids:   [ batch_size, data_dim ] np
        Returns:
        - [ batch_size ] th or np
        '''
        if isinstance(xgrids, np.ndarray):
            xgrids = torch.tensor(xgrids, requires_grad=True).float()               # [ batch_size, data_dim ] th

        pdf = self.pdf(xgrids, dtype, otype='torch')            # [ batch_size ] th
        log_p = torch.log(pdf)                                  # [ batch_size ] th
        grads = torch.autograd.grad(outputs  =   log_p,
                        inputs               =   xgrids,
                        grad_outputs=torch.ones_like(log_p),    # [ batch_size ] th, vector-Jacobian product
                        create_graph= True,
                        retain_graph= True)[0]
        
        if otype == 'numpy':
            grads = grads.detach().numpy()      # [ batch_size, data_dim ] np
        elif otype == 'torch':
            pass
        else:
            raise NotImplementedError('Unrecognized otype!')

        return grads    # [ batch_size, data_dim ] th or np
    
    def H_score(self, x, dtype = 0):
        '''
        Compute the hyvarinen score for a given piece of sample
        Args:
        - x:    [ batch_size, data_dim ] np
        Returns:
        - H:    [ batch_size ] np
        '''
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, requires_grad=True).float()
        score = self.score(x, dtype, otype = 'torch')    # [ batch_size, data_dim ] th
        grads = []
        for i in range(score.shape[-1]):
            grad = torch.autograd.grad(
                outputs = score[:, i],  # [ batch_size ] th
                inputs  = x,            # [ batch_size, data_dim ] th
                grad_outputs = torch.ones_like(score[:, i]),
                create_graph=True,
                retain_graph=True
            )[0][:, i]                  # [ batch_size ] th
            grads.append(grad)
        grads = torch.stack(grads, 1)   # [ batch_size, data_dim ] th
        div   = grads.sum(-1)           # [ batch_size ] th
        H = 1/2 * torch.norm(score, dim=1)**2 + div
        return H.detach().numpy()       # [ batch_size ] np
    
def project_line_to_full_circle(a, b, r, n_points=100, upper=True):
    '''
    Returns:
    - [ npoints, 2 ] np
    '''
    t = np.linspace(0, 1, n_points)[:-1]
    theta = 2 * np.pi * t if upper else 2 * np.pi * (1 + t)
    x = a + 2 * r * np.cos(theta)
    y = b + 2 * r * np.sin(theta)
    return np.stack([x, y], -1)


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import torch.utils
import torch.utils.data

class Dense(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Dense, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.relu  = nn.ReLU()

    def forward(self, x):
        activation = self.linear(x)
        activation = self.relu(activation)
        return activation

class DeepNetSampler(nn.Module):

    def __init__(self, latent_dim = 4, data_dim = 2, layers = [64], device = 'cpu'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # pre-change
        nn_ = [ Dense(latent_dim, layers[0]) ]
        for i in range(len(layers) - 1):
            nn_ += [ Dense(layers[i], layers[i + 1]) ]
        nn_ += [ nn.Linear(layers[-1], data_dim) ]
        self.nn0 = nn.Sequential(*nn_).to(device)

        # post-change
        nn_ = [ Dense(latent_dim, layers[0]) ]
        for i in range(len(layers) - 1):
            nn_ += [ Dense(layers[i], layers[i + 1]) ]
        nn_ += [ nn.Linear(layers[-1], data_dim) ]
        self.nn1 = nn.Sequential(*nn_).to(device)

        self.apply(self.initialize)

        self.prior = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

    def sample(self, batch_size, n0, n1):
        '''
        Returns:
        - data: [ batch_size, n0 + n1, data_dim ] th
        '''
        with torch.no_grad():
            data = self.prior.sample([batch_size, n0,]) # [ batch_size, nsample, data_dim ] th
            data0 = self.nn0(data)    # [ batch_size, n0, data_dim ] th
            data = self.prior.sample([batch_size, n1,]) # [ batch_size, nsample, data_dim ] th
            data1 = self.nn1(data)    # [ batch_size, n1, data_dim ] th
            data = torch.cat([data0, data1], 1) # [ batch_size, nsample, data_dim ] th
        return data.numpy()
    
    def fit(self, nsample, batch_size, lr, nepoch,
            lam0, lam1, lam2, save_path):
        data = self.prior.sample([nsample,])   # [ nsample, latent_dim ] th
        ds   = torch.utils.data.TensorDataset(data) 
        dl   = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True) 
        optim   =  torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for i in range(nepoch):
            losses_ = []
            for batch in dl:
                x = batch[0]
                optim.zero_grad()
                y0 = self.nn0(x)    # [ batch_size, data_dim ] th
                y1 = self.nn1(x)
                mu_diff     = torch.mean((y0 - y1)**2)
                cov_diff    = torch.mean((y0.T.cov() - y1.T.cov())**2)
                mmd         = self.mmd(y0, y1) 
                loss        = lam0 * mu_diff + lam1 * cov_diff - lam2 * mmd
                loss.backward()
                optim.step()
                losses_.append(loss.item())
            losses.append(np.mean(losses_))

            if i % (nepoch // 10) == 0:
                print(f'Epoch: {i} \t Loss: {losses[-1]}')
        torch.save(self.state_dict(), save_path)

    def load(self, save_path):
        self.load_state_dict(torch.load(save_path))
        print('Model loaded!')

    def mmd(self, X, Y, gamma = 1.0):
        '''
        Args:
        - x, y:     [ batch_size, data_dim ] th
        Returns:
        - mmd:      [ batch_size, data_dim ] th
        '''
        XX = self.rbf_kernel(X, X, gamma)
        YY = self.rbf_kernel(Y, Y, gamma)
        XY = self.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    
    @staticmethod
    def rbf_kernel(X, Y, gamma=1.0):
        """Compute the RBF kernel between two sets of data."""
        XX = torch.sum(X**2, dim=1, keepdim=True)  # Shape: [N, 1]
        YY = torch.sum(Y**2, dim=1, keepdim=True)  # Shape: [M, 1]
        XY = torch.matmul(X, Y.T)                  # Shape: [N, M]
        distances = XX - 2 * XY + YY.T             # Shape: [N, M]
        return torch.exp(-gamma * distances)
    
    @staticmethod
    def initialize(layer):
        '''
        Initialize neural network using Gaussian distrbution
        '''
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0., std=1.)
            torch.nn.init.normal_(layer.bias, mean=0., std=1.)

    @staticmethod
    def plot_ellipse(mean, cov, ax = None, n_std=2.0, **kwargs):
        '''
        numpy
        helper function --- check if the data is distint enough in terms of first and second order moments (Gaussian CUSUM) 
        '''
        if ax is None:
            fig, ax = plt.subplots()

        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

        ax.add_patch(ellipse)
        ax.scatter(*mean, color=kwargs.get('edgecolor', None))  # Plot the mean as a red point
        ax.set_xlim(mean[0] - 3*width, mean[0] + 3*width)
        ax.set_ylim(mean[1] - 3*height, mean[1] + 3*height)
        ax.set_aspect('equal')
        return ax
