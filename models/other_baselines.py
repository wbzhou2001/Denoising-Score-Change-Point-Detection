import numpy as np
from data.sampler import GaussianMixtureSampler

class GaussianCUSUM:

    def __init__(self, Mu1, Mu2, Sigma1, Sigma2):
        '''
        Args:
        1 is pre-change, 2 is post-change
        - Mu:      [ data_dim ] np
        - Sigma:   [ data_dim, data_dim ] np
        '''
        self.Mu1 = Mu1
        self.Mu2 = Mu2
        self.Sigma1 = Sigma1
        self.Sigma2 = Sigma2
        self.Sigma1_inv = np.linalg.inv(Sigma1)
        self.Sigma2_inv = np.linalg.inv(Sigma2)

    def Delta(self, x):
        '''
        Compute the Delta statistics
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        Returns:
        - [ batch_size, time_length ] np
        '''
        batch_size, time_length, data_dim = x.shape
        x_ = x.reshape(batch_size * time_length, data_dim)                                  # [ batch_size * time_length, data_dim ] np
        
        term1 = 1/2 * np.log( np.linalg.det(self.Sigma1) / np.linalg.det(self.Sigma2) )

        mini_batch_size = 256
        term2, term3 = [], []
        for i in range(len(x_) // mini_batch_size + 1):
            x__ = x_[i * mini_batch_size:(i + 1) * mini_batch_size]
            term2_ = 1/2 * ( (x__ - self.Mu1) @ self.Sigma1_inv @ (x__ - self.Mu1).T )    # [ mini_batch_size, mini_batch_size  ] np
            term3_ = - 1/2 * ( (x__ - self.Mu2) @ self.Sigma2_inv @ (x__ - self.Mu2).T )  # [ mini_batch_size, mini_batch_size ] np
            term2.append(np.diag(term2_))
            term3.append(np.diag(term3_))
        term2 = np.concatenate(term2)   # [ batch_size * time_length ] np
        term3 = np.concatenate(term3)   # [ batch_size * time_length ] np
        Delta = term1 + term2 + term3   # [ batch_size * time_length ] np
        return Delta.reshape(batch_size, time_length)

    def statistics(self, x):
        '''
        Get the statistics for each t
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        Returns:
        - [ batch_size, time_length ] np 
        '''
        batch_size, time_length, _ = x.shape
        Delta = self.Delta(x)           # [ batch_size, time_length ] np
        Ss = [ np.zeros(batch_size) ]
        for t in range(time_length):
            S = np.clip(Ss[t], a_min=0, a_max=None) + Delta[:, t]
            Ss.append(S)
        return np.stack(Ss[1:], -1)     # [ batch_size, time_length ] np 
    
    def calibrate(self, Gamma, Dcal0):
        '''
        Calibrate for the detection threshold
        Args:
        - Gamma:    scalar, desired average run length (ARL).
        - Dcal0:    [ cal_size, traj_len, data_dim ] np, calibration data pre-change
        Returns:
        - b:        scalar, threshold
        '''
        _, d2, _ = Dcal0.shape
        Scal = self.statistics(Dcal0)           # [ cal_size, traj_len ] np
        W = np.max(Scal, -1)                    # [ cal_size ] np
        b = np.quantile(W, np.exp(-d2/Gamma))   # scalar
        self.b = b
        return b

    def stopping_time(self, D):
        '''
        Args:
        - D: [ batch_size, time_length, data_dim ] np
        Returns:
        - T: [ batch_size ]
        '''
        if self.b == None:
            raise NotImplementedError("Model has not been calibrated yet!")
        S = self.statistics(D)                                  # [ batch_size, time_length ] np
        S = np.concatenate([S, np.ones([S.shape[0], 1]) * 2 * self.b], 1)   # [ batch_size, time_length + 1 ] np
        T = np.argmax(S >= self.b, axis=1)                      # [ batch_size ] np
        return T
        

    @staticmethod
    def get_mu_sigma(x):
        '''
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        '''
        x_ = x.reshape(-1, x.shape[-1])
        Mu      = x_.mean(0)                 # [ data_dim ] np
        Sigma   = np.cov(x_, rowvar=False)   # [ data_dim ] np
        return Mu, Sigma
    

class CUSUM:
    def __init__(self, sampler : GaussianMixtureSampler):
        '''
        Args:
        - sampler: sampler with access to PDF method
        '''
        self.sampler = sampler

    def Delta(self, x):
        '''
        Args:
        - x:        [ batch_size, time_length, data_dim ] np
        Returns:
        - Delta:    [ batch_size, time_length ] np
        '''
        batch_size, time_length, data_dim = x.shape
        x_ = x.reshape(-1, data_dim)
        lik_pre  = self.sampler.pdf(x_, dtype = 0)  # [ batch_size * time_length ] np
        lik_post = self.sampler.pdf(x_, dtype = 1)  # [ batch_size * time_length ] np
        Delta = np.log(lik_post) - np.log(lik_pre)
        Delta = Delta.reshape(batch_size, time_length)

        return Delta

    def statistics(self, x):
        '''
        Get the statistics for each t
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        Returns:
        - [ batch_size, time_length ] np 
        '''
        batch_size, time_length, _ = x.shape
        Delta = self.Delta(x)           # [ batch_size, time_length ] np
        Ss = [ np.zeros(batch_size) ]
        for t in range(time_length):
            S = np.clip(Ss[t], a_min=0, a_max=None) + Delta[:, t]
            Ss.append(S)
        return np.stack(Ss[1:], -1)     # [ batch_size, time_length ] np 
    
    def calibrate(self, Gamma, Dcal0):
        '''
        Calibrate for the detection threshold
        Args:
        - Gamma:    scalar, desired average run length (ARL).
        - Dcal0:    [ cal_size, traj_len, data_dim ] np, calibration data pre-change
        Returns:
        - b:        scalar, threshold
        '''
        _, d2, _ = Dcal0.shape
        Scal = self.statistics(Dcal0)           # [ cal_size, traj_len ] np
        W = np.max(Scal, -1)                    # [ cal_size ] np
        b = np.quantile(W, np.exp(-d2/Gamma))   # scalar
        self.b = b
        return b

    def stopping_time(self, D):
        '''
        Args:
        - D: [ batch_size, time_length, data_dim ] np
        Returns:
        - T: [ batch_size ]
        '''
        if self.b == None:
            raise NotImplementedError("Model has not been calibrated yet!")
        S = self.statistics(D)                                  # [ batch_size, time_length ] np
        S = np.concatenate([S, np.ones([S.shape[0], 1]) * 2 * self.b], 1)   # [ batch_size, time_length + 1 ] np
        T = np.argmax(S >= self.b, axis=1)                      # [ batch_size ] np
        return T
        

class SCUSUM:

    def __init__(self, sampler : GaussianMixtureSampler,
                 Q = None):
        '''
        Args:
        - sampler: sampler with access to PDF method
        '''
        self.sampler = sampler
        
        if Q is not None:
            self.sampler.transform_Q(Q)

    def Delta(self, x):
        '''
        Args:
        - x:        [ batch_size, time_length, data_dim ] np
        Returns:
        - Delta:    [ batch_size, time_length ] np
        '''
        batch_size, time_length, data_dim = x.shape
        x_ = x.reshape(-1, data_dim)    # [ batch_size * time_length, data_dim ] np
        H_pre  = self.sampler.H_score(x_, dtype = 0)  # [ batch_size * time_length ] np
        H_post = self.sampler.H_score(x_, dtype = 1)  # [ batch_size * time_length ] np
        H_pre, H_post = H_pre.reshape(batch_size, time_length), H_post.reshape(batch_size, time_length) # [ batch_size, time_length ] np
        Delta = H_pre - H_post  # [ batch_size, time_length ] np
        return Delta

    def statistics(self, x):
        '''
        Get the statistics for each t
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        Returns:
        - [ batch_size, time_length ] np 
        '''
        batch_size, time_length, _ = x.shape
        Delta = self.Delta(x)           # [ batch_size, time_length ] np
        Ss = [ np.zeros(batch_size) ]
        for t in range(time_length):
            S = np.clip(Ss[t], a_min=0, a_max=None) + Delta[:, t]
            Ss.append(S)
        return np.stack(Ss[1:], -1)     # [ batch_size, time_length ] np 
    
    def calibrate(self, Gamma, Dcal0):
        '''
        Calibrate for the detection threshold
        Args:
        - Gamma:    scalar, desired average run length (ARL).
        - Dcal0:    [ cal_size, traj_len, data_dim ] np, calibration data pre-change
        Returns:
        - b:        scalar, threshold
        '''
        _, d2, _ = Dcal0.shape
        Scal = self.statistics(Dcal0)           # [ cal_size, traj_len ] np
        W = np.max(Scal, -1)                    # [ cal_size ] np
        b = np.quantile(W, np.exp(-d2/Gamma))   # scalar
        self.b = b
        return b

    def stopping_time(self, D):
        '''
        Args:
        - D: [ batch_size, time_length, data_dim ] np
        Returns:
        - T: [ batch_size ]
        '''
        if self.b == None:
            raise NotImplementedError("Model has not been calibrated yet!")
        S = self.statistics(D)                                  # [ batch_size, time_length ] np
        S = np.concatenate([S, np.ones([S.shape[0], 1]) * 2 * self.b], 1)   # [ batch_size, time_length + 1 ] np
        T = np.argmax(S >= self.b, axis=1)                      # [ batch_size ] np
        return T

# TODO: fix the above derivations

from sklearn.mixture import GaussianMixture
class GMCUSUM:

    def __init__(self,
                 ncomp_pre, ncomp_post):
        self.model_pre  = GaussianMixture(n_components = ncomp_pre, random_state=0)
        self.model_post = GaussianMixture(n_components = ncomp_post, random_state=0)

    def fit(self, Dpre, Dpost):
        '''
        Args:
        - D: [ batch_size, time_length, data_dim ] np
        '''
        self.model_pre.fit(Dpre.reshape(-1, Dpre.shape[-1]))
        self.model_post.fit(Dpost.reshape(-1, Dpost.shape[-1]))

    def Delta(self, x):
        '''
        Args:
        - x:        [ batch_size, time_length, data_dim ] np
        Returns:
        - Delta:    [ batch_size, time_length ] np
        '''
        batch_size, time_length, data_dim = x.shape
        x_ = x.reshape(-1, data_dim)    # [ batch_size * time_length, data_dim ] np
        LL_pre  = self.model_pre.score_samples(x_)
        LL_post = self.model_post.score_samples(x_)
        Delta = LL_post - LL_pre
        Delta = Delta.reshape(batch_size, time_length)  # [ batch_size, time_length ] np
        return Delta

    def statistics(self, x):
        '''
        Get the statistics for each t
        Args:
        - x: [ batch_size, time_length, data_dim ] np
        Returns:
        - [ batch_size, time_length ] np 
        '''
        batch_size, time_length, _ = x.shape
        Delta = self.Delta(x)           # [ batch_size, time_length ] np
        Ss = [ np.zeros(batch_size) ]
        for t in range(time_length):
            S = np.clip(Ss[t], a_min=0, a_max=None) + Delta[:, t]
            Ss.append(S)
        return np.stack(Ss[1:], -1)     # [ batch_size, time_length ] np 
    
    def calibrate(self, Gamma, Dcal0):
        '''
        Calibrate for the detection threshold
        Args:
        - Gamma:    scalar, desired average run length (ARL).
        - Dcal0:    [ cal_size, traj_len, data_dim ] np, calibration data pre-change
        Returns:
        - b:        scalar, threshold
        '''
        _, d2, _ = Dcal0.shape
        Scal = self.statistics(Dcal0)           # [ cal_size, traj_len ] np
        W = np.max(Scal, -1)                    # [ cal_size ] np
        b = np.quantile(W, np.exp(-d2/Gamma))   # scalar
        self.b = b
        return b

    def stopping_time(self, D):
        '''
        Args:
        - D: [ batch_size, time_length, data_dim ] np
        Returns:
        - T: [ batch_size ]
        '''
        if self.b == None:
            raise NotImplementedError("Model has not been calibrated yet!")
        S = self.statistics(D)                                  # [ batch_size, time_length ] np
        S = np.concatenate([S, np.ones([S.shape[0], 1]) * 2 * self.b], 1)   # [ batch_size, time_length + 1 ] np
        T = np.argmax(S >= self.b, axis=1)                      # [ batch_size ] np
        return T
    
'''
Online models
'''

from models.base_cusum import OnlineCUSUM 
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist 
from tqdm import tqdm

class GMCUSUM(OnlineCUSUM):

    def __init__(self, w, ncomp):
        '''
        Args:
        - pre_save_path:    save path of the pre-change score model's state dict
        '''
        super().__init__(w)
        self.model1 = GaussianMixture(n_components=ncomp, random_state=0)
        self.model2 = GaussianMixture(n_components=ncomp, random_state=0)

    def fit(self, Dtr0):
        '''
        Args:
        -   [ batch_size, data_dim ] np
        '''
        self.model1.fit(Dtr0)
        self.model2.fit(Dtr0)
        self.Dtr0 = Dtr0

    def reset(self):
        self.model1.fit(self.Dtr0)
        self.model2.fit(self.Dtr0)

    def update(self, Dtr):
        self.model2.fit(Dtr)

    def delta(self, x):
        '''
        Single statistics
        Args:
        - x:        [ data_dim ] np
        Returns:
        - delta:    scalar
        '''
        x = x.reshape(1, -1) # [ 1, data_dim ] np
        delta = self.model1.score_samples(x) - self.model2.score_samples(x)
        return delta.item()
    
class MStatistics:

    def __init__(self, B, N, Dtr, sigma):
        '''
        By default uses the Gaussian kernel
        
        Args:
        - Dtr0: [ tr_seq_len, data_dim ] np
        - sigma: bandwidth of the Gaussian kernel
        '''
        self.Dref = Dtr
        # Dtr[-N*B:].reshape(N, B, -1) # [ N, B, data_dim ] np, reference dataset
        self.B = B
        self.N = N
        self.sigma = sigma
        pass

    def statistics(self, Dte):
        '''
        Args:
        - Dte:  [ te_seq_len, data_dim ]
        '''
        S = [ 0 for _ in range(self.B) ]
        Dref = np.concatenate(
            [self.Dref.copy(), Dte[:self.B]], axis=0
        )       # [ tr_seq_len + B , data_dim ] np, reference dataset
        for i in tqdm(range(self.B, len(Dte))):
            indices = np.random.choice(len(Dref), self.B * self.N)
            Xref    = Dref[indices].reshape(self.N, self.B, -1)     # [ N, B, data_dim ] np
            X       = Dte[i-self.B:i]                               # [ B, data_dim ] np
            M = 0.
            for j in range(self.N):
                M += self.MMD(Xref[j], X)
            S.append(M / self.N)
            Dref    = np.concatenate([Dref, Dte[[i]]], axis=0)          # [ seq_len ++ , data_dim ] np
        return S

    def MMD(self, x, y):
        '''
        Args:
        - x:    [ b1, data_dim ] np
        - y:    [ b2, data_dim ] np
        Returns:
        -       scalar
        '''
        xx = self.gaussian_kernel(x, x) # [ b1, b1 ] np
        yy = self.gaussian_kernel(y, y) # [ b2, b2 ] np
        xy = self.gaussian_kernel(x, y) # [ b1, b2 ] np
        val = xx[~np.eye(xx.shape[0]).astype(bool)].mean() + yy[~np.eye(yy.shape[0]).astype(bool)].mean() - 2 * xy.mean()
        return val

    def gaussian_kernel(self, x, y):
        '''
        Args:
        - x:    [ b1, data_dim ] np
        - y:    [ b2, data_dim ] np
        Retruns:
        -       [ b1, b2 ] np
        '''
        dist = cdist(x, y, metric='euclidean')  # [ b1, b2 ] np
        val = np.exp( - dist / 2 * self.sigma**2)
        return val
    
class HotellingT:

    def __init__(self, Dtr):
        '''[ tr_seq_len, data_dim ] np'''
        self.mu  = np.mean(Dtr, axis=0)  # [ data_dim ]
        self.cov = np.cov(Dtr.T)         # [ data_dim, data_dim ]
        
    def statistics(self, Dte):
        '''[ te_seq_len, data_dim ] np'''
        T = (Dte - self.mu) @ self.cov @ (Dte - self.mu).T  # [ te_seq_len, te_seq_len ] np
        T = np.diag(T)
        return T    # [ te_seq_len ] np