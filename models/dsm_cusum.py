import numpy as np
import torch
from models.score_matching import ScoreMatching, DenoisingScoreMatching

class OfflineScoreMatchingCUSUM:
    '''Score Matching Cumulative Sum Control Chart (Offline Version)'''
    def __init__(self,
                 init_kwargs,
                 type = 'hyvarinen'):
        if type == 'hyvarinen':
            self.model1 = ScoreMatching(**init_kwargs)
            self.model2 = ScoreMatching(**init_kwargs)
        elif type == 'denoising':
            self.model1 = DenoisingScoreMatching(**init_kwargs)
            self.model2 = DenoisingScoreMatching(**init_kwargs)
        else:
            raise NotImplementedError
        self.b = None

    def fit(self,
            data1, data2,
            fit_kwargs):
        '''
        Fit the two models
        Args:
        - data1:    [ batch_size, data_dim ] th
        - data2:    [ batch_size, data_dim ] th
        '''
        fit_kwargs1, fit_kwargs2 = fit_kwargs.copy(), fit_kwargs.copy()
        fit_kwargs1['save_path'] = fit_kwargs1['save_path'] + '/model_1'
        fit_kwargs2['save_path'] = fit_kwargs2['save_path'] + '/model_2'
        self.model1.fit(torch.tensor(data1), **fit_kwargs1)
        print('Model 1 training completed!')
        self.model2.fit(torch.tensor(data2), **fit_kwargs2)
        print('Model 2 training completed!')

    def load(self, save_path):
        self.model1.load(save_path + '/model_1')
        self.model2.load(save_path + '/model_2')
    
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
        Scal = self.statistics(Dcal0)            # [ cal_size, traj_len ] np
        W = np.max(Scal, -1)                    # [ cal_size ] np
        b = np.quantile(W, np.exp(-d2/Gamma))   # scalar
        self.b = b
        return b

    def stopping_time(self, D):
        '''
        Args:
        - D:    [ batch_size, time_length, data_dim ] np
        Returns:
        - T:    [ batch_size ] np
        '''
        if self.b == None:
            raise NotImplementedError("Model has not been calibrated yet!")
        S = self.statistics(D)                  # [ batch_size, time_length ] np
        S = np.concatenate([S, np.ones([S.shape[0], 1]) * 2 * self.b], 1)   # [ batch_size, time_length + 1 ] np
        T = np.argmax(S >= self.b, axis=1)      # [ batch_size ] np
        return T
    
    def statistics(self, D, batch_size_ = 512):
        '''
        Args:
        - D:            [ batch_size, time_length, data_dim ] np
        - batch_size_:  batch_size used for 
        Returns:
        - S:    [ batch_size, time_length ] np
        '''
        self.model1.eval()
        self.model2.eval()
        batch_size, time_length, data_dim  = D.shape
        S = [ torch.zeros(batch_size) ]
        D = D.reshape(-1, data_dim)                                     # [ batch_size * time_length, data_dim ] np
        
        S_ = []
        for i in range(len(D) // batch_size_ + 1):
            D_ = D[i * batch_size_ : (i + 1) * batch_size_]
            s_difs_ = self.model1.H_score(D_) - self.model2.H_score(D_) # [ batch_size * time_length ] np
            S_.append(s_difs_)
        s_difs = np.concatenate(S_, 0)                                  # [ batch_size * time_length ] np

        s_difs = s_difs.reshape(batch_size, time_length)                # [ batch_size, time_length ] th
        for t in range(time_length):                                    # time_length iterations
            Stp1 = np.clip(S[-1], a_min=0, a_max=None) + s_difs[:, t]    # [ batch_size ] th     
            S.append(Stp1)
        S = np.stack(S, -1)   # [ batch_size, time_length ] th
        return S


from models.base_cusum import OnlineCUSUM
class OnlineDenoisingScoreMatchingCUSUM(OnlineCUSUM):

    def __init__(self, w, init_kwargs, update_kwargs,
                 save_path):
        '''
        Args:
        - pre_save_path:    save path of the pre-change score model's state dict
        '''
        super().__init__(w)
        self.model1 = DenoisingScoreMatching(**init_kwargs)
        self.model2 = DenoisingScoreMatching(**init_kwargs)
        self.save_path = save_path
        self.update_kwargs = update_kwargs

    def fit(self, Dtr0, fit_kwargs):
        self.model1.fit(Dtr0, **fit_kwargs, save_path=self.save_path)
        self.model2.load(self.save_path)

    def reset(self):
        self.model1.load(self.save_path)
        self.model2.load(self.save_path)

    def update(self, Dtr):
        self.model2.fit(Dtr, **self.update_kwargs, verbose = False) # TODO

    def delta(self, x):
        '''
        Single statistics
        Args:
        - x:        [ data_dim ] np
        Returns:
        - delta:    scalar
        '''
        x = x.reshape(1, -1) # [ 1, data_dim ] np
        delta = self.model1.H_score(x) - self.model2.H_score(x)
        return delta.item()