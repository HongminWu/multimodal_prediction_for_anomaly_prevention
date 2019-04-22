import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import ipromps
import ipdb
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter1d            

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = "Times New Roman"

def run(group_errors_by_csv = None,
        group_stamps_by_csv = None,
        c = None):
    figuresize = (12, 3)
    fig, ax = plt.subplots(figsize=figuresize)
    plt.subplots_adjust(bottom=0.2)
    csvs = group_errors_by_csv.keys()
    usecsvs = []
    window = 5
    for i, csv in enumerate(csvs):
        error = group_errors_by_csv[csv][:,0]
        if error.shape[-1] < 90: continue
        _filtered = gaussian_filter1d(error, sigma=0.5)
        # series = pd.Series(data=error)
        # _filtered = series.rolling(window=window).mean()[window:].values
        ax.plot(_filtered, label='Error sequence')
        usecsvs.append(csv)
        
    plt.xlabel("Duration")
    plt.ylabel("Error")
    plt.xlim(0,210)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=1)
    fig.savefig('./figures/error_sequences_w_diff_lengths.png', format="png", papertype='a4')

    
    #--------------------------------------
    csvs = usecsvs
    lengths = [group_errors_by_csv[csv].shape[0] for csv in csvs]
    regular_length = int(round(np.average(lengths)))
    durations = [group_stamps_by_csv[csv][-1] for csv in csvs] # unit: seconds

    # align the time with regular_length
    values_stack = None
    group_errors_by_csv_aligned = {}
    for i, csv in enumerate(csvs):
        stamp_raw = np.linspace(0, durations[i], lengths[i])
        stamp_aligned = np.linspace(0, stamp_raw[-1], regular_length)
        values_filtered = group_errors_by_csv[csv]
        values_aligned = griddata(stamp_raw, values_filtered, stamp_aligned, method='linear')
        group_errors_by_csv_aligned[csv] = values_aligned
        if values_stack is None:
            values_stack = values_aligned
        else:
            values_stack = np.vstack([values_stack, values_aligned])
    
    n_features = group_errors_by_csv_aligned[csvs[0]].shape[-1]

    noise_cov = np.cov(values_stack.T)
    
    ipromp = ipromps.IProMP(num_joints = n_features,
                            num_obs_joints = n_features,
                            num_basis=35,
                            sigma_basis=0.05,
                            num_samples=regular_length,
                            sigmay= noise_cov,
                            min_max_scaler= None,
                              )

    for i, csv in enumerate(csvs):
        data = group_errors_by_csv_aligned[csv]
        ipromp.add_demonstration(data)
        ipromp.add_alpha(durations[i])
            
    fig, ax = plt.subplots(figsize=figuresize)
    plt.subplots_adjust(bottom=0.2)    
    plt.axes(ax)
    ipromp.promps[0].plot_prior(b_regression=False, linewidth_mean=3, b_dataset=True, c = c)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=1)
    plt.xlabel("Temporal Scalar")
    plt.ylabel("Error")
    plt.xlim(0,1)
    fig.savefig('./figures/error_sequences_model.png', format="png", papertype='a4')
    plt.show()    

if __name__=="__main__":
    group_errors_by_csv = np.load('group_errors_by_csv.npy').item()
    group_stamps_by_csv = np.load('group_stamps_by_csv.npy').item()
    run(group_errors_by_csv = group_errors_by_csv,
        group_stamps_by_csv = group_stamps_by_csv,
            c = 3.0)
