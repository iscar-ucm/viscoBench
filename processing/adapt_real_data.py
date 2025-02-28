"""
    Usage examples:

        python adapt_real_data.py "data/validation/Ethanol_70pc" 1000

        python adapt_real_data.py "data/validation/IPA_100pc" 1000 1500

"""

import pandas as pd 
import numpy as np
import sys, os
import matplotlib.pyplot as plt

from scipy import interpolate


class user_inputer:
    def __init__(self) -> None:
        self.t_pairs = []
        self.filenames = []
        self.last_t_pair = []
        self.current_filename = ""
        self.y_min = 0.0
        self.y_max = 2000
        self.lines_c = ["g", "r"]
        
    def onclick(self, event):
        """
        Event tracker for the plot. It is used to catch the actions from
        the user s.t. the signal can be split in the right sequences.
        """
        if len(self.last_t_pair) < 2:
            self.last_t_pair.append(event.xdata)

            plt.vlines(event.xdata, ymin=self.y_min, ymax=self.y_max, 
                       colors=self.lines_c[len(self.last_t_pair)-1])
            plt.draw()
        if len(self.last_t_pair) == 2:
            self.t_pairs[-1].append( self.last_t_pair )
            self.filenames.append( self.current_filename )
            self.last_t_pair = []
        print( "t: {:.3f}s".format(event.xdata) )


def adapt_data_format():
    if len(sys.argv) < 3:
        print("Data folder and out sampling points must be specified.")
        exit()
    datapath = sys.argv[1]

    n_out_points = []
    for i in range(2, len(sys.argv)):
        n_out_points.append( int(sys.argv[i]) )
    X = [[] for _ in range(len(n_out_points))]
    Y = [[] for _ in range(len(n_out_points))]
    incT = [[] for _ in range(len(n_out_points))]
    data_cols = ["Time [s]", "P1(Read)[mbar]", "P2(Read)[mbar]"]
    u_inputer = user_inputer()
    
    print("Getting data and interpolating sequences to {} points.".format(n_out_points))

    # Gather all the files' names
    files = []
    for (dirpath, dirnames, filenames) in os.walk(datapath):
        files.extend(filenames)
    print("Found files at {}: {}".format(datapath, files))
    # Load the viscoties CSV
    visc_filename = ""
    for fname in files:
        if ".csv" in fname:
            visc_filename = fname
            break
    if visc_filename == "":
        print("Missing viscosities CSV file in specified folder: {}".
              format(datapath))
        exit()
    
    # From each file, split the signals
    visc_df = pd.read_csv(datapath + "/" + visc_filename)
    for idx, row in visc_df.iterrows():
        # Compose the name of the file
        filename = row["signal"]+ "_" + \
                   row["p_range"]+ "_" + \
                   "{}C.txt".format(row["avg_temp"])
        # Update user_inputer values
        u_inputer.current_filename = filename
        u_inputer.t_pairs.append( [] )
        sigs_df = pd.read_csv(datapath + "/" + filename, sep="\t")
        u_inputer.y_min = min(
            sigs_df[data_cols[1]].min(),
            sigs_df[data_cols[2]].min()
        )
        u_inputer.y_max = max(
            sigs_df[data_cols[1]].max(),
            sigs_df[data_cols[2]].max()
        )

        # Interactive plot
        fig, ax = plt.subplots()
        plt.plot(sigs_df[data_cols[0]], sigs_df[data_cols[1]], label="P1")
        plt.plot(sigs_df[data_cols[0]], sigs_df[data_cols[2]], label="P2")
        plt.legend()
        plt.grid(True)
        fig.canvas.mpl_connect('button_press_event', u_inputer.onclick)
        plt.show()

        # Get the data into the right format
        t = sigs_df[data_cols[0]].tolist()
        p1 = sigs_df[data_cols[1]].tolist()
        p2 = sigs_df[data_cols[2]].tolist()
        for t_pair in u_inputer.t_pairs[-1]:
            # Get the index of closest t (for start and end)
            idx_start = min(range(len(t)), key=lambda i: abs(t[i]-t_pair[0]))
            idx_end = min(range(len(t)), key=lambda i: abs(t[i]-t_pair[1]))
            # Interpolate data
            f1 = interpolate.interp1d(t[idx_start:idx_end], p1[idx_start:idx_end])
            f2 = interpolate.interp1d(t[idx_start:idx_end], p2[idx_start:idx_end])
            for idx, n_points in enumerate(n_out_points):
                itp_t = np.linspace(t[idx_start], t[idx_end-1], n_points)
                # Append the signals to a numpy array as well as viscosity and period
                X[idx].append( np.transpose([f1(itp_t), f2(itp_t)]) )
                Y[idx].append( row["viscosity"] )
                incT[idx].append( itp_t[1]-itp_t[0] )
    
    # Save the data as numpy X and Y matrices
    for idx, n_points in enumerate(n_out_points):
        foldername = "/processed_{}pt".format(n_points)
        if not os.path.exists(datapath + foldername):
            os.mkdir( datapath + foldername )

        with open(datapath + foldername + "/metadata.txt", "w") as f: 
            f.write("Origin folder: {}\n".format(datapath))
            f.write("N samples: {}\n".format(n_points))
        np.save(datapath + foldername + "/X_samples.npy", X[idx])
        np.save(datapath + foldername + "/Y_samples.npy", Y[idx])
        np.save(datapath + foldername + "/incT_samples.npy", incT[idx])


if __name__ == "__main__":
    adapt_data_format()