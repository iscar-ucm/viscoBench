"""
    Usage example:

    python check_synthetic_dataset.py "data/dataset01_full_500pt_12s"
    
"""

import numpy as np
import yaml

import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Paths
    if len(sys.argv) != 2:
        print("Data folder must be specified.")
        exit()
    datapath = sys.argv[1] # "data/dataset02_segments_500samples"

    # Load the training data
    X = np.load(datapath+"/X_samples.npy")
    Y = np.load(datapath+"/Y_samples.npy")

    print("Dimensions X: {}".format(np.shape(X)))
    print("Dimensions Y: {}\n".format(np.shape(Y)))

    # Load the configuration
    cfg_path = "cfg/training_cfg.yaml"
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    X_norm = np.multiply(np.divide(X, np.subtract(cfg["max_X"], cfg["min_X"])) ,
                         cfg["norm_X_range"]) + cfg["norm_X_offset"]
    Y_norm = (Y / (cfg["max_Y"] - cfg["min_Y"])) \
        * cfg["norm_Y_range"] + cfg["norm_Y_offset"]
    # Y_norm = np.transpose( np.transpose(Y_norm)[cfg["out_col"]] )

    # Print normalization data
    print("In min vals: {}".format(cfg["min_X"]))
    print("In max vals: {}".format(cfg["max_X"]))
    print("Out min vals: {}".format(cfg["min_Y"]))
    print("Out max vals: {}\n".format(cfg["max_Y"]))

    print("Min\max from X: {}, {}".format(np.min(X), np.max(X)))
    print("Min\max from Y: {}, {}".format(np.min(np.transpose(np.transpose(Y)[cfg["out_col"]])), 
                                          np.max(np.transpose(np.transpose(Y)[cfg["out_col"]]))))
    print("Min\max from X norm: {}, {}".format(np.min(X_norm), np.max(X_norm)))
    print("Min\max from Y norm: {}, {}\n".format(np.amin(Y_norm), np.amax(Y_norm)))

    print("Dimensions X_norm: {}".format(np.shape(X_norm)))
    print("Dimensions Y_norm: {}\n".format(np.shape(Y_norm)))

    ans = "y"
    s_i = 0
    while (ans != "n") and (ans != "N"):
        X_plot = X
        X_plot_norm = X_norm
        Y_plot = Y
        Y_plot_norm = Y_norm
        # Keep plotting until user inserts "n" or "N"
        sorted = False
        if not sorted:
            s_i = np.random.randint(len(X_plot))
        else: 
            s_i += 1

        P1_i = np.transpose(X_plot[s_i])[0]
        P2_i = np.transpose(X_plot[s_i])[1]
        t_i = np.transpose(X_plot[s_i])[2]
        y_i = Y_plot[s_i]
        P1_i_norm = np.transpose(X_plot_norm[s_i])[0]
        P2_i_norm = np.transpose(X_plot_norm[s_i])[1]
        t_i_norm = np.transpose(X_plot_norm[s_i])[2]
        y_i_norm = Y_plot_norm[s_i]

        plt.subplot(2,1,1)
        plt.plot(t_i, P1_i)
        plt.ylabel("P")
        plt.grid()

        plt.subplot(2,1,1)
        plt.plot(t_i, P2_i)

        plt.subplot(2,1,2)
        plt.plot(t_i_norm, P1_i_norm)
        plt.ylabel("P_norm")

        plt.subplot(2,1,2)
        plt.plot(t_i_norm, P2_i_norm)
        plt.grid()

        y_desn = ((y_i_norm - cfg["norm_Y_offset"]) / cfg["norm_Y_range"]) * \
            (cfg["max_Y"] - cfg["min_Y"])

        print("For sample {}, output Y: {}, Y_norm: {}, Y_desnorm: {}".format(
            s_i, y_i, y_i_norm, y_desn
        ))
        plt.show()

        ans = input("Another sample? [y/n]: ")
