import sys
import numpy as np

import pywt
from scipy import interpolate

import matplotlib.pyplot as plt


class db5_signal:
    def __init__(self, th_psi=1e-3) -> None:
        self.__name__ = "db5_signal"
        self.wavelet = pywt.Wavelet("db5")
        phi, psi, x = self.wavelet.wavefun(level=4)
        # Remove silent space off signal
        for idx in range(len(psi)): # From the beginning
            if psi[idx] > th_psi:
                fist_i = idx 
                break
        for idx in range(len(psi)-1, 0, -1): # From the end
            if psi[idx] > th_psi:
                last_i = idx 
                break
        self.psi = psi[fist_i:last_i]
        self.t = x[fist_i:last_i]
        self.kT = self.t[-1] - self.t[0] # Func Period
        # Interpolation function
        self.db5_fn = interpolate.interp1d(self.t, self.psi)

    def __call__(self, t, A=1.0, k=1.0):
        # Ensure that t is within period time
        t_periodic = t*k
        while t_periodic < self.t[0]:
            t_periodic += self.kT
        while t_periodic > self.t[-1]:
            t_periodic -= self.kT
        # Call the interpolation function
        return A*(self.db5_fn(t_periodic) + 1)/2
    
if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Arguments for daubechies-5 must be specified.")
        exit()
    A = float(sys.argv[1])
    k = float(sys.argv[2])

    db5 = db5_signal()
    t = np.linspace(0, 20, 100)

    y = []
    for t_i in t: 
        y.append( db5(t_i, A, k) )

    print("Len: {}, from x0: {} to xN-1: {}".format(len(t), t[0], t[-1]))
    plt.plot(t, y)
    plt.show()