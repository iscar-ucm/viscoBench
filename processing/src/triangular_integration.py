import sys
import numpy as np

from scipy import signal

import matplotlib.pyplot as plt


class triangular_signal:
    def __init__(self, A=1.0, k=1.0) -> None:
        self.__name__ = "triangular_signal"
        self.A = A
        self.k = k
        
    def __call__(self, t):
        return self.A*(signal.sawtooth(2*np.pi*self.k*t*0.1, 0.5)+1.0) / 2
    
if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Arguments for triangular signal must be specified.")
        exit()
    A = float(sys.argv[1])
    k = float(sys.argv[2])

    triang = triangular_signal(A, k)
    t = np.linspace(0, 12, 500)

    y = []
    for t_i in t: 
        y.append( triang(t_i) )

    print("Len: {}, from x0: {} to xN-1: {}".format(len(t), t[0], t[-1]))
    plt.plot(t, y)
    plt.show()