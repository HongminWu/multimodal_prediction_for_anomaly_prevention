# achieve the anomaly detection by combining the LSTM and ProMPs,
# where the LSTM for prdicting the time series and the ProMPs for modeling the predicting errors for each dimention(feature), respectively

from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import cPickle as pickle
import ipdb, os, glob
from collections import OrderedDict
from sklearn import preprocessing
from anomaly_detection_with_prediction import (detect_anomalies_with_prediction_actuals,
                                              plot_anomaly_with_matplotlib)
from rnn_models import (get_model_via_name, series_to_supervised)
import ipromps 
from scipy.interpolate import griddata
import random

def convert_to_X_y(dataset = None, n_lags=None):
    values = dataset.values
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(values)

    # data is sampling in 50HZ specify the number of lags = 10
    n_features = values.shape[-1]
    # frame as supervised learning
    reframed = series_to_supervised(train_scaled, n_lags, 1)

    # split into train and test sets
    refValues = reframed.values
    
    # split into input and outputs
    n_obs = n_lags * n_features
    X, y = refValues[:, :n_obs], refValues[:, -n_features:]

    # reshape input to be 3D [samples, timesteps, features]
    X = X.reshape((X.shape[0], n_lags, n_features))
    
    return X, y, scaler, n_features

def model_errors_with_prob_by_feature(group_errors_by_csv = None, group_stamps_by_csv = None): 
    csvs = group_errors_by_csv.keys()
    lengths = [group_errors_by_csv[csv].shape[0] for csv in csvs]
    regular_length = int(round(np.average(lengths)))
    durations = [group_stamps_by_csv[csv][-1] for csv in csvs] # unit: seconds

    # align the time with regular_length
    group_errors_by_csv_aligned = {}
    for i, csv in enumerate(csvs):
        stamp_raw = np.linspace(0, durations[i], lengths[i])
        stamp_aligned = np.linspace(0, stamp_raw[-1], regular_length)
        values_filtered = group_errors_by_csv[csv]
        values_aligned = griddata(stamp_raw, values_filtered, stamp_aligned, method='linear')
        group_errors_by_csv_aligned[csv] = values_aligned
    
    n_features = group_errors_by_csv_aligned[csvs[0]].shape[-1]
    
    group_promp_model_by_feature = OrderedDict()
    for feature in range(n_features):
        min_max_scaler = preprocessing.MinMaxScaler()
        promp = ipromps_lib.ProMP(num_basis=35,
                                  sigma_basis=0.05,
                                  num_samples=regular_length,
                                  sigmay=0.0,
                                  min_max_scaler=None,
                                  )
        print('processing dim=%d'%feature)
        for i, csv in enumerate(csvs):
            data = group_errors_by_csv_aligned[csv][:,feature]
            promp.add_demonstration(data)
            promp.add_alpha(durations[i])
            
        group_promp_model_by_feature[feature] = promp

    return group_promp_model_by_feature


def status(e = None, mean =  None, std = None, c = None):
    upperThreshold = mean + c * std
    lowerThreshold = mean - c * std
    if e > upperThreshold or e < lowerThreshold :
        return "anomalies"
    else:
        return "norminal"


def testing(csv, n_lags, prediction_model, promp_models, anomaly_t_by_human = None, c = None, succ = None):
    dataset = read_csv(csv, header=0, index_col=0)
    x, y, scaler, n_features = convert_to_X_y(dataset = dataset, n_lags = n_lags)
    input_shape = (x.shape[1], x.shape[2])

    # make a prediction
    yhat = prediction_model.predict(x)
    # invert scaling for forecasting
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    inv_y = scaler.inverse_transform(y)

    fig, axarr = plt.subplots(nrows=n_features, ncols=1, sharex=True)
    plt.subplots_adjust(hspace=0.5)

    dtype = "succ" if succ else "unsucc"
    
    plt.suptitle("%s: %s"%(dtype, os.path.basename(csv)))

    stamp = (dataset.index - dataset.index[0])[:-n_lags]

    if anomaly_t_by_human is not None:
        anomaly_time = anomaly_t_by_human -  dataset.index[0]
    
    for feature in range(n_features):
        print "feature: %s"%feature
        _yhat = inv_yhat[:,feature]
        _y = inv_y[:,feature]
        e = _y - _yhat

        print "shape of e: %s" %e.shape

        plt.axes(axarr[feature])
        axarr[feature].set_title('probabilistic model of feature #%s'%feature)        

        means, stds = promp_models[feature].plot_prior(b_regression=False, linewidth_mean=3, b_dataset=True, c = c)
        # testing the new trail
        alpha_candi = promp_models[feature].alpha_candidate(num_alpha_candidate = 10)
        idx_max = promp_models[feature].estimate_alpha(alpha_candi, e, stamp)
        print('alpha_candidate_max_idx=%d'%idx_max)
        estimated_alpha = alpha_candi[idx_max]['candidate']
        promp_models[feature].set_alpha(estimated_alpha)
        print('Adding testing points in each promp model...')

        if anomaly_t_by_human is not None:
            axarr[feature].axvline(anomaly_time / estimated_alpha, c='black', ls = '--')
        
        for idx in range(len(stamp)):
            t = stamp[idx] / estimated_alpha

            if t > 1.0: t = 0.998

            promp_models[feature].add_viapoint(t, e[idx])
            try:
                mean = promp_models[feature].get_mean(t)
            except Exception as exp:
                ipdb.set_trace()
            ix = np.where(means == mean)[0][0]
            std = stds[ix]
            if status(e = e[idx], mean = mean, std = std, c = c) == "anomalies":
                print "triggered @ %s"%stamp[idx]
        promp_models[feature].plot_uViapoints()
    
def split_testing(succ_csvs, unsucc_csvs, n_lags, prediction_model, promp_models):

    c = 5.0
    
    for csv in succ_csvs:
        print
        print "Ganna to test the successful trials one by one"
        print
        testing(csv, n_lags, prediction_model, promp_models, anomaly_t_by_human = None, c = c, succ=True)
        
    for csv in unsucc_csvs:
        print
        print "Ganna to test the unsuccessful trials one by one"
        print
        anomaly_label_and_signal_time = pickle.load(open(os.path.join(
            os.path.dirname(csv),
            'anomaly_label_and_signal_time.pkl'
        ), 'r'))
        anomaly_type = anomaly_label_and_signal_time[0]
        anomaly_t_by_human = anomaly_label_and_signal_time[1].to_sec()
       
        testing(csv, n_lags, prediction_model, promp_models, anomaly_t_by_human = anomaly_t_by_human, c = c, succ=False)
 
    plt.show()
            
if __name__=="__main__":
    data_path =  "/home/birl_wu/baxter_ws/src/SPAI/smach_based_introspection_framework/temp_folder_prediction_for_error_prevention_wrench_norm/anomaly_detection_feature_selection_folder/No.0 filtering scheme"
    skill = 3
    sampling_rate = 50 # hz
    
    # load successful dataset for training and validating
    succ_csvs = glob.glob(os.path.join(
        data_path,
        'successful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))

    train_succ_csvs = random.sample(succ_csvs, int(len(succ_csvs)/3)) # 1/3 for training, 2/3 for validating
    valid_succ_csvs = set(succ_csvs) - set(train_succ_csvs) 
    
    unsucc_csvs = glob.glob(os.path.join(
        data_path,
        'unsuccessful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))
    
    n_lags = 10
    train_X = None
    train_y = None
    for csv in train_succ_csvs:
        train_dataset = read_csv(csv, header=0, index_col=0)
        X, y, scaler, n_features = convert_to_X_y(dataset = train_dataset, n_lags=n_lags)
        if train_X is None:
            train_X = X
            train_y = y
        else:
            train_X = np.vstack((train_X, X))
            train_y = np.vstack((train_y, y))            
    
    valid_X = None
    valid_y = None
    for csv in valid_succ_csvs:
        valid_dataset = read_csv(csv, header=0, index_col=0)
        X, y, scaler, n_features = convert_to_X_y(dataset = valid_dataset, n_lags=n_lags)
        if valid_X is None:
            valid_X = X
            valid_y = y
        else:
            valid_X = np.vstack((valid_X, X))
            valid_y = np.vstack((valid_y, y))            

    input_shape = (train_X.shape[1], train_X.shape[2])
    model_name = "lstm" #"lstm_dropout" "lstm" 
    model = get_model_via_name(model_name= model_name,
                               input_shape = input_shape,
                               output_shape = n_features)

    # fit network
    _weights = "models/%s_model.h5"%model_name
    _history = "models/%s_history.pickle"%model_name    
    if not os.path.isfile(_weights):
        history = model.fit(train_X,
                            train_y,
                            epochs=100,
                            batch_size=1,
                            validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
        model.save_weights(_weights)
        with open(_history, "wb") as f:
            pickle.dump(history, f)
            
        # plot history
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        
    else:
        print ("Model trained, loading the weights and history...")
        model.load_weights(_weights)
        with open(_history, "rb") as f:
            history = pickle.load(f)
    
    fig, axarr = plt.subplots(nrows=n_features, ncols=1, sharex=False)
    plt.subplots_adjust(hspace=0.5)
    group_errors_by_csv = OrderedDict()
    group_stamps_by_csv = OrderedDict() 
    for i, csv in enumerate(valid_succ_csvs):
        valid_dataset = read_csv(csv, header=0, index_col=0)
        valid_X, valid_y, _, _ = convert_to_X_y(dataset = valid_dataset, n_lags=n_lags)

        # make a prediction
        yhat   = model.predict(valid_X)
        # invert scaling for forecasting
        inv_yhat = scaler.inverse_transform(yhat)
        # invert scaling for actual
        inv_y = scaler.inverse_transform(valid_y)

        for feature in range(n_features):
            _yhat = inv_yhat[:,feature]
            _y = inv_y[:,feature]
            # calculate RMSE
            rmse = sqrt(mean_squared_error(_y,_yhat))
            print('Test RMSE: %.3f of feature_%s' % (rmse, feature))

            # plot the error by actual - predict
            e = _y - _yhat
            axarr[feature].plot(e, label='errors' if i==0 else "")            
            axarr[feature].legend()
            axarr[feature].set_title('error sequences of feature #%s'%feature)
        fileID = "csv-%s"%i
        group_errors_by_csv[fileID] = inv_y - inv_yhat
        group_stamps_by_csv[fileID] = valid_dataset.index - valid_dataset.index[0]
    print group_errors_by_csv.keys()

    # build probabilistic model for each feature
    group_promp_model_by_feature = model_errors_with_prob_by_feature(group_errors_by_csv = group_errors_by_csv,
                                                                     group_stamps_by_csv = group_stamps_by_csv,
                                                                     )
    
    split_testing(succ_csvs = valid_succ_csvs,
                    unsucc_csvs = unsucc_csvs,
                    n_lags = n_lags,
                    prediction_model = model,
                    promp_models = group_promp_model_by_feature,
                        )
    plt.show()
