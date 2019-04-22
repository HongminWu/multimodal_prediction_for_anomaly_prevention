import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = "Times New Roman"
errors = np.load('errors.npy')
fig, ax = plt.subplots(figsize=(12,3))

for i, error in enumerate(errors):
    if error.shape[-1] < 90: continue

    series = pd.Series(data=error)
    mean = series.rolling(window=5).mean()
    ax.set_xlim(0,220)
    ax.plot(mean)
    
plt.xlabel("Duration")
plt.ylabel("Error")
fig.savefig('error_sequences_w_diff_lengths.png', format="png", dpi=300)
plt.show()    
