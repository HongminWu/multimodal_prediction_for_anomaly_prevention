import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

def plot_anomaly_with_matplotlib(df, anomaly_t_by_human = None, anomaly_type='success', filename = 'default'):
    st = df['test_time'].values[0]
    t  = df['test_time'].values - st
    anomaly_t_human = anomaly_t_by_human - st
    bool_array = (abs(df['anomaly_points']) > 0)
    #And a subplot of the Actual Values.
    actuals = df["actuals"][-len(bool_array):]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan
    adf = df.dropna(subset=['anomaly_points'])
    anomaly_times = adf['test_time'].values - st
    fig, axarr = plt.subplots(nrows=2, ncols=1)
    
    # subplot-1: errors
    axarr[0].plot(t, df['error'], c='red', label='predicted error')
    axarr[0].plot(t, df['meanval'], c='black', lw = 3, label='rolling mean')    
    axarr[0].fill_between(t, df['3s'], df['-3s'], facecolor='yellow', alpha=0.5, label = 'safety')
    axarr[0].scatter(anomaly_times, adf['anomaly_points'], c = 'red', s=30,  label='anomalies')
    axarr[0].axvline(anomaly_t_human,  c = 'black', ls = '--', label='occurrence')    
    axarr[0].legend(loc=1)
    
    # subplot-2: actuals
    xmin, xmax = axarr[1].get_xlim()
    ymin, ymax = axarr[1].get_ylim()
    axarr[1].plot(t, df['actuals'], c = 'blue', label="actuals")
    axarr[1].plot(t, df['predicted'], c = 'orange', label="predicted")
    axarr[1].axvline(anomaly_t_human,  c = 'black', ls = '--', label='occurrence')
    axarr[1].scatter(anomaly_times, anomaly_points.dropna(), c = 'red', s=30, label='anomalies')
    axarr[1].text(anomaly_t_human, ymax-0.05*(ymax-ymin), anomaly_type, color='red', fontsize=12, rotation=-90)    
    axarr[1].legend(loc=1)
    plt.title(filename)
    fig.savefig('./figures/%s.png'%filename, format = 'png')

def detect_anomalies_with_prediction_actuals(df, window = 20):
    '''
        Steps for detecting anomalies:
        1. Compute the error term(actual- predicted).
        2. Compute the rolling mean and rolling standard deviation(window is a week).
        3. Classify data with an error of 1.5,1.75 and 2 standard deviations as limits for low,medium and high anomalies. 
    '''
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error']=df['actuals']-df['predicted']
    df['percentage_change'] = ((df['actuals'] - df['predicted']) / df['actuals']) * 100
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()

    print ("@Hongmin Wu: fill the NaN values as the first point")
    df['meanval'].fillna(df['meanval'][window-1], inplace=True) 
    df['deviation'].fillna(df['deviation'][window-1], inplace=True)   
    
    print ("finished caculating the error, mean and std")
    print 
    
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                               range(len(df['error']))]

    print ("fininshed definied the impacts of errors")
    
    severity = {0: 3,
                1: 2,
                2: 1,
                3: 0,
                4: 0,
                5: 1,
                6: 2,
                7: 3}
        
    df['color'] =  df['impact'].map(severity)
    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan) #Return elements chosen from x or y depending on condition.
    
    return df
