import pandas as pd 
import numpy as np
import sys, os
import matplotlib.pyplot as plt


class user_inputer:
    def __init__(self) -> None:
        self.t_pairs = []
        self.filenames = []
        self.last_t_pair = []
        self.current_filename = ""
        
    def onclick(self, event):
        """
        Interactive plot from user @armatita on Stackoverflow: 
        https://stackoverflow.com/questions/37363755/
        python-mouse-click-coordinates-as-simply-as-possible
        """
        if len(self.last_t_pair) < 2:
            self.last_t_pair.append(event.xdata)
        if len(self.last_t_pair) == 2:
            self.t_pairs.append( self.last_t_pair )
            self.filenames.append( self.current_filename )
            self.last_t_pair = []
        print( "t: {:.3f}s".format(event.xdata) )


def adapt_data_format():
    if len(sys.argv) != 2:
        print("A folder containing the data must be specified.")
        exit()
    datapath = sys.argv[1]
    data_cols = ["Time [s]", "P1(Read)[mbar]", "P2(Read)[mbar]"]
    u_inputer = user_inputer()
    X, Y = [], []

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
        u_inputer.current_filename = filename
        sigs_df = pd.read_csv(datapath + "/" + filename, sep="\t")
        print(sigs_df)

        # Interactive plot
        fig, ax = plt.subplots()
        plt.plot(sigs_df[data_cols[0]], sigs_df[data_cols[1]], label="P1")
        plt.plot(sigs_df[data_cols[0]], sigs_df[data_cols[2]], label="P2")
        plt.legend()
        fig.canvas.mpl_connect('button_press_event', u_inputer.onclick)
        plt.show()

        # Append the signals to a numpy array
        # Append the label of the signal (viscosity)
    
    # Normalize the data
    # Save the data as numpy X and Y matrices


if __name__ == "__main__":
    adapt_data_format()