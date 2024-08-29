import numpy as np
from scipy import signal


class constant_signal:
    def __init__(self) -> None:
        self.__name__ = "constant_signal"

    def __call__(self, t, A=1.0, k=1.0):
        return A


class sinusoidal_signal:
    def __init__(self) -> None:
        self.__name__ = "sinusoidal_signal"
        
    def __call__(self, t, A=1.0, k=1.0):
        return A*np.sin(k*t)


class triangular_signal:
    def __init__(self) -> None:
        self.__name__ = "triangular_signal"
        
    def __call__(self, t, A=1.0, k=1.0):
        return A*(signal.sawtooth(2*np.pi*k*t, 0.5)+1.0) / 2