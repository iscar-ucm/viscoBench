import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


# Experimental Benchmark X - X := Number of input samples
eb500_path = "benchmarks/experimental_500pt.csv"
eb1000_path = "benchmarks/experimental_1000pt.csv"
eb1500_path = "benchmarks/experimental_1500pt.csv"
# Synthetic Benchmark X - X := Number of input samples
sb500_path = "benchmarks/synthetic_500pt.csv"
sb1000_path = "benchmarks/synthetic_1000pt.csv"
sb1500_path = "benchmarks/synthetic_1500pt.csv"


if __name__=="__main__":
    # Load the data
    # Experimental
    eb500_df = pd.read_csv(eb500_path)
    eb1000_df = pd.read_csv(eb1000_path)
    eb1500_df = pd.read_csv(eb1500_path)
    ebs = [eb500_df, eb1000_df, eb1500_df]
    # Synthetic
    sb500_df = pd.read_csv(sb500_path)
    sb1000_df = pd.read_csv(sb1000_path)
    sb1500_df = pd.read_csv(sb1500_path)
    sbs = [sb500_df, sb1000_df, sb1500_df]

    # Group the data by algorithm
    benchmark_dict = {alg: {"exp_acc": [], 
                            "synt_acc_clean": [],
                            "synt_acc_noise": [],
                            "exp_time": [], 
                            "synt_time_clean": [],
                            "synt_time_noise": []} 
                            for alg in eb500_df["algorithm"]}
    for alg in benchmark_dict.keys():
        for b in sbs:
            # Synthetic data without noise
            mean_err = b.loc[ (b["algorithm"] == alg) & (b["noise"] == 0.0) ]["rel_error"].mean()
            mean_runtime = b.loc[ (b["algorithm"] == alg) & (b["noise"] == 0.0) ]["runtime"].mean()*1000
            benchmark_dict[alg]["synt_acc_clean"].append(
                max(0, 100 - mean_err)
            )
            benchmark_dict[alg]["synt_time_clean"].append( mean_runtime )
            # Synthetic data with noise
            mean_err = b.loc[ (b["algorithm"] == alg) & (b["noise"] > 0.0) ]["rel_error"].mean()
            mean_runtime = b.loc[ (b["algorithm"] == alg) & (b["noise"] > 0.0) ]["runtime"].mean()*1000
            benchmark_dict[alg]["synt_acc_noise"].append(
                max(0, 100 - mean_err)
            )
            benchmark_dict[alg]["synt_time_noise"].append( mean_runtime )
        for b in ebs:
            # Experimental data
            benchmark_dict[alg]["exp_acc"].append(
                max(0, 100 - b.loc[b["algorithm"] == alg]["rel_error"].iloc[0])
            )
            benchmark_dict[alg]["exp_time"].append( b.loc[b["algorithm"] == alg]["runtime"].iloc[0]*1000 )
    
    # Plot the data
    styles = {
        "N4SID": { "color": "k", "linestyle": "--", "marker": "o"},
        "FROLS": { "color": "b", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "x"},
        "LSTM": { "color": "c", "linestyle": (0, (5, 10)), "marker": "*"},
        "CNN": { "color": "g", "linestyle": ":", "marker": "^"},
        "qLSTM": { "color": "m", "linestyle": "-.", "marker": "v"},
        "qCNN": { "color": "r", "linestyle": "-", "marker": "D"},
    }
    resolutions = [500, 1000, 1500]
    ylim = [38, 100]

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 2])

    # Create the top three subplots (spanning the first two rows)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[:2, 2])
    # Create the bottom three subplots
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])

    # Without Noise
    ax1.grid(True)
    ax1.set_title("$\mathcal{Synth.}, \mathcal{N}_{\sigma}=\mathcal{0.000}$")
    for alg in benchmark_dict.keys():
        ax1.plot(resolutions, benchmark_dict[alg]["synt_acc_clean"], label=alg, **styles[alg])
    ax1.set_ylabel("Accuracy [%]")
    ax1.set_xticks(resolutions)
    ax1.set_ylim(*ylim)
    ax1.legend()

    # With Noise
    ax2.set_title("$\mathcal{Synth.}, \mathcal{N}_{\sigma}=\mathcal{0.025}$")
    ax2.grid(True)
    for alg in benchmark_dict.keys():
        ax2.plot(resolutions, benchmark_dict[alg]["synt_acc_noise"], label=alg, **styles[alg])
    ax2.set_xticks(resolutions)
    ax2.set_ylim(*ylim)
    
    # Experimental Data
    ax3.set_title("$\mathcal{Exp.}$")
    ax3.grid(True)
    for alg in benchmark_dict.keys():
        ax3.plot(resolutions, benchmark_dict[alg]["exp_acc"], label=alg, **styles[alg])
    ax3.set_xticks(resolutions)
    ax3.set_ylim(*ylim)

    # And runtimes!
    ax4.grid(True)
    for alg in benchmark_dict.keys():
        ax4.plot(resolutions, benchmark_dict[alg]["synt_time_clean"], label=alg, **styles[alg])
    ax4.set_xticks(resolutions)
    ax4.set_ylabel("Inference Time [ms]")

    ax5.grid(True)
    for alg in benchmark_dict.keys():
        ax5.plot(resolutions, benchmark_dict[alg]["synt_time_noise"], label=alg, **styles[alg])
    ax5.set_xticks(resolutions)
    ax5.set_xlabel("Number of Input Samples [n.u.]")

    ax6.grid(True)
    for alg in benchmark_dict.keys():
        ax6.plot(resolutions, benchmark_dict[alg]["exp_time"], label=alg, **styles[alg])
    ax6.set_xticks(resolutions)

    plt.show()

    for alg in benchmark_dict.keys():
        print("{}. Inference times: {}".format(alg, benchmark_dict[alg]["synt_time_clean"]))