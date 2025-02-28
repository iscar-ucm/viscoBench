import numpy as np
import pywt
from scipy import signal, interpolate


class step_signal:
    def __init__(self, A=1.0, k=1.0, offset=0.0) -> None:
        self.__name__ = "step_signal"
        self.A = A
        self.k = k
        self.offset = offset

    def __call__(self, t):
        res = np.sin(self.k*t*np.pi)
        return self.A if res > 0.0 else self.offset


class sinusoidal_signal:
    def __init__(self, A=1.0, k=1.0, offset=0.0) -> None:
        self.__name__ = "sinusoidal_signal"
        self.A = A
        self.k = k
        self.offset = offset
        
    def __call__(self, t):
        return self.A*(np.sin(self.k*t*np.pi)+1)/2 + self.offset


class triangular_signal:
    def __init__(self, A=1.0, k=1.0, offset=0.0) -> None:
        self.__name__ = "triangular_signal"
        self.A = A
        self.k = k
        self.offset = offset
        
    def __call__(self, t):
        return (self.A*(signal.sawtooth(self.k*t*np.pi*2, 0.5)+1.0) / 2
                + self.offset)

class db5_signal:
    def __init__(self, th_psi=1e-3, A=1.0, k=1.0, offset=0.0) -> None:
        self.__name__ = "db5_signal"
        self.A = A
        self.k = k
        self.offset = offset
        # Define the wavelet function
        self.wavelet = pywt.Wavelet("db5")
        phi, psi, x = self.wavelet.wavefun(level=4)
        # Remove silent space off signal
        for idx in range(len(psi)): # From the beginning
            if np.abs(psi[idx]) > th_psi:
                fist_i = idx 
                break
        for idx in range(len(psi)-1, 0, -1): # From the end
            if np.abs(psi[idx]) > th_psi:
                last_i = idx 
                break
        self.psi = psi[fist_i:last_i]
        self.t = x[fist_i:last_i]
        self.kT = self.t[-1] - self.t[0] # Func Period
        # Interpolation function
        self.db5_fn = interpolate.interp1d(self.t, self.psi)

    def __call__(self, t):
        # Ensure that t is within period time
        t_periodic = t*self.k*np.pi*2.5
        while t_periodic < self.t[0]:
            t_periodic += self.kT
        while t_periodic > self.t[-1]:
            t_periodic -= self.kT
        # Call the interpolation function
        return self.A*(self.db5_fn(t_periodic)+1)/2 + self.offset