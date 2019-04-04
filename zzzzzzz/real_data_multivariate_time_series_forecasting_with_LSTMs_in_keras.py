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

from anomaly_detection_with_time_series_forcasting import (detect_anomalies_with_prediction_actuals,
                                                               plot_anomaly_with_plotly)

from rnn_models import (get_model_via_name, series_to_supervised)


if __name__=="__main__":
    '''
    # load successful dataset for training and validating
    basic_path = "/home/birl_wu/keras_tensorflow_learning/"
    skill = 3
    succ_folders = glob.glob(os.path.join(
        basic_path,
        'successful_skills',
        'skill %s'%skill,
    )) 
    '''
    
    train_data_path = "/home/birl_wu/keras_tensorflow_learning/successful_skills/skill 3/No.0_successful_skill_from_experiment_at_2018y05m19d15H48M03S/skill3.csv"
    
    #test_data_path = "/home/birl_wu/keras_tensorflow_learning/successful_skills/skill 3/No.0_successful_skill_from_experiment_at_2018y05m24d21H48M25S/skill3.csv"

    test_data_path = "/home/birl_wu/keras_tensorflow_learning/unsuccessful_skills/skill 3/No.0_unsuccessful_skill_from_experiment_at_2018y05m24d21H47M19S/skill3.csv"
    anomaly_label_and_signal_time = pickle.load(open(os.path.join(
        os.path.dirname(test_data_path),
        'anomaly_label_and_signal_time.pkl'
    ), 'r'))
    anomaly_type = anomaly_label_and_signal_time[0]
    anomaly_t_by_human = anomaly_label_and_signal_time[1].to_sec()
    
    train_dataset = read_csv(train_data_path, header=0, index_col=0)
    test_dataset = read_csv(test_data_path, header=0, index_col=0)
    
    train_dataset.plot(sharex=True, subplots = True, title='raw data of train dataset')
    train_date = pd.to_datetime(train_dataset.index.values, unit='s')
    test_date = pd.to_datetime(test_dataset.index.values, unit='s')    
    train_values = train_dataset.values
    test_values = test_dataset.values    

    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_values)
    test_scaled  = scaler.fit_transform(test_values)    

    # data is sampling in 50HZ specify the number of lags = 10
    n_lags = 10
    n_features = train_values.shape[-1]
    # frame as supervised learning
    train_reframed = series_to_supervised(train_scaled, n_lags, 1)
    test_reframed = series_to_supervised(test_scaled, n_lags, 1)    

    print 'train_shape:'
    print(train_reframed.shape)
    print 'test_shape:'
    print(test_reframed.shape)    

    # split into train and test sets
    train = train_reframed.values
    test = test_reframed.values
    
    # split into input and outputs
    n_obs = n_lags * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features:]
    test_X, test_y = test[:, :n_obs], test[:, -n_features:]
    print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model_name = "lstm"
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = get_model_via_name(model_name= model_name, input_shape = input_shape, output_shape = n_features)

    # for saving
    _weights = "%s_model.h5"%model_name
    _history = "%s_history.pickle"%model_name

    # fit network
    if not os.path.isfile(_weights):
        history = model.fit(train_X,
                            train_y,
                            epochs=100,
                            batch_size=1,
                            validation_data=(test_X, test_y), verbose=2, shuffle=False)
        model.save_weights(_weights)
        with open(_history, "wb") as f:
            pickle.dump(history, f)
    else:
        print "Model trained, loading the weights and history..."
        model.load_weights(_weights)
        with open(_history, "rb") as f:
            history = pickle.load(f)

    # plot history
    pyplot.figure()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()

    # make a prediction
    yhat   = model.predict(test_X)
    # invert scaling for forecasting
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    inv_y = scaler.inverse_transform(test_y)
  
    fig, axarr = pyplot.subplots(nrows=n_features, ncols=1, sharex=True)
    for dim in range(n_features):
        _yhat = inv_yhat[:,dim]
        _y = inv_y[:,dim]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(_y,
                                       _yhat))
        print('Test RMSE: %.3f of dim_%s' % (rmse, dim))

        # plot the predict and actual
        axarr[dim].plot(_y, label='actuals')
        axarr[dim].plot(_yhat, label='predicted')

        # detect anomalies
        predicted_df = DataFrame()
        predicted_df['load_date'] = test_date[n_lags:]
        predicted_df['actuals'] =  _y
        predicted_df['predicted']= _yhat
        classify_df = detect_anomalies_with_prediction_actuals(predicted_df, window = n_lags)
        classify_df.reset_index(inplace=True)
        del classify_df['index']
        plot_anomaly_with_plotly(classify_df, anomaly_t_by_human = anomaly_t_by_human, metric_name = "Time series prediction for anomaly detection", filename = "un_dim_%s"%dim)
        
    pyplot.legend()
    pyplot.show()
