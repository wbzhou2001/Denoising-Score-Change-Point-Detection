
import numpy as np
import torch
from tqdm import tqdm

class BaseCUSUM:

    def __init__(self):
        pass

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

    def manual_set_b(self, b):
        self.b = b

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

class OfflineCUSUM(BaseCUSUM):

    def __init__(self):
        super().__init__()


    def delta(self, x):
        '''
        Single statistics
        Args:
        - x:    [ batch_size, seq_len, data_dim ] np
        Returns:
        - S:    [ batch_size, seq_len ] np
        '''
        pass
    
    def statistics(self, x):
        '''
        Detection statistics
        Args:
        - x:    [ batch_size, seq_len, data_dim ] np
        Returns:
        - S:    [ batch_size, seq_len ] np
        '''
        batch_size, seq_len, data_dim = x.shape 
        deltas = self.delta(x)      # [ batch_size, seq_len ] np
        S = []
        for i in range(seq_len):
            delta = deltas[:, i]    # [ batch_size ] np
            stp1 = np.clip(S[-1], a_min=0, a_max=None) + delta
            S.append(stp1)
        S = np.stack(S, -1)         # [ batch_size, seq_len ] np
        return S

class OnlineCUSUM(BaseCUSUM):

    def __init__(self, w):
        super().__init__()
        self.w = w

    def delta(self, x):
        '''
        Single statistics
        Args:
        - x:    [ data_dim ] np
        Returns:
        - S:    scalar
        '''
        pass

    def reset(self):
        pass

    def update(self, Dtr):
        '''
        Args:
        - Dtr: [ w, data_dim ]
        '''
        # NOTE: pre-change and post-change score should be initialized to be the same.
        pass
    
    def statistics(self, X):
        '''
        NOTE: this algorithm is highly resource inefficient
        Detection statistics
        Args:
        - X:    [ batch_size, seq_len, data_dim ] np
        Returns:
        - S:    [ seq_len ] np
        '''
        batch_size, seq_len, data_dim = X.shape
        S_ = []
        for j in range(batch_size):
            self.reset()
            x = X[j]
            S = [0 for i in range(self.w)]
            for i in tqdm(range(self.w, seq_len), leave=False):
                Dtr = x[i - self.w:i]
                self.update(Dtr)
                delta = self.delta(x[i])
                stp1 = np.clip(S[-1], a_min=0, a_max=None) + delta
                S.append(stp1)
            S = np.stack(S, -1) # [ seq_len ] np
            S_.append(S)
        S_ = np.stack(S_, 0)      # [ batch_size, seq_len ] np
        return S_