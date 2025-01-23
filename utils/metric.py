from tqdm import tqdm
import numpy as np

def WADD(model, D):
    '''
    Test function for experimental evaluation
    Args:
    - D: test Monte Carlo distibution
    '''
    # TODO: assuming that the model has already been calibrated
    t       = model.stopping_time(D)  # [ n_iter ] np
    WADD    = t.mean()
    std     = t.std() 
    return WADD, std, t, D

class WADDClass:

    def __init__(self, sg,
                 xxx, yyy, n_iter, len_max,
                 start, end):
        self.ARLs = 10 ** np.linspace(start, end, 10)
        self.Dcal0  = sg.sample(xxx, yyy, 0)
        self.D      = sg.sample(n_iter, 0, len_max)

    def WADD(self, model):
        X, Y, Y_stds, Bs = [], [], [], []
        for Gamma in tqdm(self.ARLs, desc='ARLs'):

            # calibration
            model.calibrate(Gamma=Gamma, Dcal0=self.Dcal0)

            x = Gamma
            y, y_std, _, _ = WADD(model, self.D)

            X.append(x)
            Y.append(y)
            Y_stds.append(y_std)
            Bs.append(model.b)
        return X, Y, Y_stds, Bs