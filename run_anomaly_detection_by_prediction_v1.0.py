# achieve the anomaly detection by the time series prediction, this implementation is copied from the website
# Anomaly Detection with Time Series Forecasting
# https://towardsdatascience.com/anomaly-detection-with-time-series-forecasting-c34c6d04b24a

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
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

from anomaly_detection_with_prediction import (detect_anomalies_with_prediction_actuals,
                                              plot_anomaly_with_matplotlib)
from rnn_models import (get_model_via_name, series_to_supervised)

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


if __name__=="__main__":

    curr_path =  os.path.dirname(os.path.realpath(__file__))
    skill = 3
    # load successful dataset for training and validating
    succ_csvs = glob.glob(os.path.join(
        curr_path,
        'successful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))

    # load unsuccessful dataset for testing
    unsucc_csvs = glob.glob(os.path.join(
        curr_path,
        'unsuccessful_skills',
        'skill %s'%skill,
        '*',
        '*.csv'
    ))
    
    train_dataset = read_csv(succ_csvs[0], header=0, index_col=0)
    valid_dataset = read_csv(succ_csvs[1], header=0, index_col=0)
    
#    train_dataset.plot(sharex=True, title='raw data of train dataset')
     
    test_dataset = read_csv(unsucc_csvs[0], header=0, index_col=0)
    anomaly_label_and_signal_time = pickle.load(open(os.path.join(
        os.path.dirname(unsucc_csvs[0]),
        'anomaly_label_and_signal_time.pkl'
    ), 'r'))
    anomaly_type = anomaly_label_and_signal_time[0]
    anomaly_t_by_human = anomaly_label_and_signal_time[1].to_sec()
    test_time = test_dataset.index.values
    
    n_lags = 10

    train_X, train_y,scaler, n_features = convert_to_X_y(dataset = train_dataset, n_lags=n_lags) 
    valid_X, valid_y, _, _  = convert_to_X_y(dataset = valid_dataset, n_lags=n_lags)
    test_X, test_y, _, _  = convert_to_X_y(dataset = test_dataset, n_lags=n_lags)       
    
    model_name = "gru_dropout" #"lstm_dropout" "lstm" 
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = get_model_via_name(model_name= model_name,
                               input_shape = input_shape,
                               output_shape = n_features)

    # for saving
    _weights = "%s_model.h5"%model_name
    _history = "%s_history.pickle"%model_name

    # fit network
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
        pyplot.figure()
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        
    else:
        print ("Model trained, loading the weights and history...")
        model.load_weights(_weights)
        with open(_history, "rb") as f:
            history = pickle.load(f)
            
    # make a prediction
    yhat   = model.predict(test_X)
    # invert scaling for forecasting
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    inv_y = scaler.inverse_transform(test_y)
  
    #fig, axarr = pyplot.subplots(nrows=n_features, ncols=1, sharex=True)
    
    for dim in range(n_features):
        _yhat = inv_yhat[:,dim]
        _y = inv_y[:,dim]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(_y,
                                       _yhat))
        print('Test RMSE: %.3f of dim_%s' % (rmse, dim))

        # plot the predict and actual
        #axarr[dim].plot(_y, label='actuals')
        #axarr[dim].plot(_yhat, label='predicted')

        # detect anomalies
        predicted_df = DataFrame()
        predicted_df['test_time'] = test_time[n_lags:]         
        predicted_df['actuals'] =  _y
        predicted_df['predicted']= _yhat
        classify_df = detect_anomalies_with_prediction_actuals(predicted_df, window = 6) # n_lags
        classify_df.reset_index(inplace=True)
        del classify_df['index']
        plot_anomaly_with_matplotlib(classify_df, anomaly_t_by_human = anomaly_t_by_human,
                                     anomaly_type = anomaly_type,
                                     filename = "%s_un_dim_%s" %(model_name, dim))
        
    pyplot.legend()
    pyplot.show()
