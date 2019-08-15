# achieve the anomaly detection by combining the LSTM and ProMPs,
# where the LSTM for prdicting the time series and the NDProMPs for modeling the predicting errors for all dimentions(features), syn.
# the threshold is considered for each feature individually
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import cPickle as pickle
import ipdb, os, glob, sys
from collections import OrderedDict
from sklearn import preprocessing
from anomaly_detection_with_prediction import (detect_anomalies_with_prediction_actuals,
                                              plot_anomaly_with_matplotlib)
from rnn_models import (get_model_via_name, series_to_supervised)
import ipromps 
from scipy.interpolate import griddata
import random
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"

GENERATE_FIGURE_IN_PAPER = False
GENERATE_FIGURE_IN_TESTING = False
dpi = 100
_labels = [r'$n_f$', r'$n_m$', r'$n_l$', r'$n_a$', r'$s_l$', r'$s_r$',]    

def convert_to_X_y(dataset = None, n_lags=None):
    values = dataset.values
    # normalize features
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
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

def model_errors_with_prob_by_features(group_errors_by_csv = None, group_stamps_by_csv = None): 
    csvs = group_errors_by_csv.keys()
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
    
    min_max_scaler = preprocessing.MinMaxScaler()
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
            
    return ipromp, durations

    
def isAnomaly(e = None, mean =  None, std = None, c = None):
    upperThreshold = mean + c * std
    lowerThreshold = mean - c * std
    if e > upperThreshold or e < lowerThreshold :
        return True
    else:
        return False


def testing(csv, n_lags, prediction_model, promp_models, anomaly_t_by_human = None, c = None, skill=None):
    dataset = read_csv(csv, header=0, index_col=0)
    x, y, scaler, n_features = convert_to_X_y(dataset = dataset, n_lags = n_lags)
    input_shape = (x.shape[1], x.shape[2])

    # make a prediction
    yhat = prediction_model.predict(x)
    # invert scaling for forecasting
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    inv_y = scaler.inverse_transform(y)

    stamp = (dataset.index - dataset.index[0])[:-n_lags]

    if anomaly_t_by_human is not None:
        anomaly_time = anomaly_t_by_human -  dataset.index[0]
        dtype = "unsucc"        
    else:
        dtype = "succ"
    
    e = inv_y - inv_yhat

    print (e.shape)
    
    # testing the new trail
    num_alpha_candidate = 6
    alpha_candi = ipromp_model.alpha_candidate( num = num_alpha_candidate)
    idx_max = ipromp_model.estimate_alpha(alpha_candi, e, stamp)
    # print('alpha_candidate_max_idx=%d'%idx_max)
    estimated_alpha = alpha_candi[idx_max]['candidate']
    ipromp_model.set_alpha(estimated_alpha)
    # print('Adding testing points in each promp model...')

    if GENERATE_FIGURE_IN_TESTING:
        fig, axarr = plt.subplots(nrows=n_features, ncols=1, sharex=False, figsize=(6, 7), tight_layout=True)
        plt.xlabel('Temporal scalar')
        fileName = "%s: %s, c=%s"%(dtype, os.path.basename(csv)[-25:], c)

    flag_of_anomalies = []
    for feature, promp in enumerate(ipromp_model.promps):
        means, stds = promp.get_means_and_stds()    
        promp.clear_viapoints()
        for idx in range(len(stamp)):
            t = stamp[idx] / estimated_alpha
            if t > 1.0: t = 0.998
            promp.add_viapoint(t, e[idx, feature])
            try:
                mean = promp.get_mean(t)
            except Exception as exp:
                ipdb.set_trace()
            ix = np.where(means == mean)[0][0]
            std = stds[ix]
            
            flag_of_anomalies.append(isAnomaly(e = e[idx, feature], mean = mean, std = std, c = c))

        if GENERATE_FIGURE_IN_TESTING:
            axarr[feature].set_title('probabilistic model of %s'%_labels[feature])
            axarr[feature].set_xlim(0,1) 
            axarr[feature].set_ylabel('Error')
            if feature < (len(ipromp_model.promps) -1 ): axarr[feature].set_xticks([])            
            plt.axes(axarr[feature])
            promp.plot_prior(b_regression=False, linewidth_mean=2, b_dataset=True, c = c)
            promp.plot_uViapoints()
            # if anomaly_t_by_human is not None:
            #     axarr[feature].axvline(anomaly_time / estimated_alpha, c='black', ls = '--')
            # handles, labels = plt.gca().get_legend_handles_labels()
            # by_label = OrderedDict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys())

    if GENERATE_FIGURE_IN_TESTING:
        fig.align_labels()
        fig.savefig("./figures/skill%s_%s.png"%(skill, fileName), format = "png", dpi=dpi)    
        
    return flag_of_anomalies, stamp[-1], estimated_alpha

def testing_results(succ_csvs, unsucc_csvs, n_lags, prediction_model, ipromp_model, c=None, skill = None):

    list_of_stamps = []
    list_of_estimated_alpha = []    
    list_of_csv_type = []
    
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    
    print ("Ganna to test the unsuccessful trials one by one, total of %s"%len(unsucc_csvs))
    for csv in unsucc_csvs: # Positive
        anomaly_label_and_signal_time = pickle.load(open(os.path.join(
            os.path.dirname(csv),
            'anomaly_label_and_signal_time.pkl'
        ), 'r'))
        anomaly_type = anomaly_label_and_signal_time[0]
        anomaly_t_by_human = anomaly_label_and_signal_time[1].to_sec()
       
        flag_of_anomalies, stamp, estimated_alpha = testing(csv, n_lags, prediction_model,
                                                               ipromp_model, anomaly_t_by_human = anomaly_t_by_human,
                                                               c = c, skill=skill)
        if sum(flag_of_anomalies): # not equal to zero, have anomalies
            TP += 1
        else:
            FN += 1
        list_of_stamps.append(stamp)
        list_of_estimated_alpha.append(estimated_alpha)
        list_of_csv_type.append('unsucc')
        
    print ("Ganna to test the successful trials one by one, total of %s"%len(succ_csvs))    
    for csv in succ_csvs: # Negative
        flag_of_anomalies, stamp, estimated_alpha = testing(csv, n_lags, prediction_model,
                                                               ipromp_model, anomaly_t_by_human = None,
                                                               c = c, skill=skill)
        if sum(flag_of_anomalies): # not equal to zero, have anomalies
            FP += 1
        else:
            TN += 1
        list_of_stamps.append(stamp)
        list_of_estimated_alpha.append(estimated_alpha)
        list_of_csv_type.append('succ')
        
    # Sensitivity and specificity
    precision = TP/(TP+FP) 
    recall    = TP/(TP+FN) # positive (abnormal) identification
    f_score   = 2*TP/(2*TP+FP+FN)
    accuracy  = (TP+TN)/(TP+TN+FP+FN)
    tpr = recall # positive (abnormal) identification
    fpr = FP / (FP + TN) # negative (nominal) identification
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
    print("TP : {:0.1f}, TN : {:0.1f}, FP : {:0.1f}, FN : {:0.1f}".format( TP, TN, FP, FN))
    print("TPR : {:0.2f}, FPR : {:0.2f}".format(tpr, fpr))
    return tpr, fpr, list_of_stamps, list_of_estimated_alpha, list_of_csv_type
    
if __name__=="__main__":
    data_path =  "./anomaly_detection_feature_selection_folder/No.0 filtering scheme"
    skill = 3
    
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
    model_name = "lstm_dropout" #"lstm_dropout" "lstm" 
    model = get_model_via_name(model_name= model_name,
                               input_shape = input_shape,
                               output_shape = n_features)

    # fit network
    _weights = "models/%s_model_skill_%s.h5"%(model_name, skill)
    _history = "models/%s_history_skill_%s.pickle"%(model_name, skill)
    if not os.path.isfile(_weights):
        history = model.fit(train_X,
                            train_y,
                            epochs=100,
                            batch_size=16,
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
            
#-----------------------Figure-1----------------        
    if GENERATE_FIGURE_IN_PAPER:
        fig, axarr = plt.subplots(nrows=n_features, ncols=2, sharex=False, figsize=(12,9))
        axarr[n_features-1, 0].set_xlabel("Duration (s)")
        axarr[n_features-1, 1].set_xlabel("Temporal scalar")        
        plt.suptitle("Error sequences with various lengths (left) and its probabilistic model with c=3.0 (right)")
        plt.subplots_adjust(top=0.91, hspace=0.3,bottom=0.1)
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
        fileID = "csv-%s"%i
        group_errors_by_csv[fileID] = inv_y - inv_yhat
        group_stamps_by_csv[fileID] = valid_dataset.index - valid_dataset.index[0]

        if GENERATE_FIGURE_IN_PAPER:
            for feature in range(n_features):
                _yhat = inv_yhat[:,feature]
                _y = inv_y[:,feature]
                # calculate RMSE
                # rmse = sqrt(mean_squared_error(_y,_yhat))
                # print('Test RMSE: %.3f of feature_%s' % (rmse, feature))
                # plot the error by actual - predict
                e = _y - _yhat
                axarr[feature,0].plot(np.linspace(0, e.shape[-1], e.shape[-1])*0.1, e)
                axarr[feature,0].set_ylabel("Error")
                axarr[feature,0].set_xlim(left=0)                
                axarr[feature,0].set_title('%s'%_labels[feature])
                if feature < (n_features - 1): axarr[feature,0].set_xticks([])

    # build probabilistic model for all features
    ipromp_model, durations = model_errors_with_prob_by_features(group_errors_by_csv = group_errors_by_csv,
                                                      group_stamps_by_csv = group_stamps_by_csv,
                                                      )
    
    if GENERATE_FIGURE_IN_PAPER:
        for feature, promp in enumerate(ipromp_model.promps):
            plt.axes(axarr[feature, 1])
            axarr[feature, 1].set_title('%s'%_labels[feature])            
            promp.plot_prior(b_regression=False, linewidth_mean=2, b_dataset=True, c = 3.0)
            axarr[feature,1].set_xlim(0, 1)
            if feature < (n_features - 1): axarr[feature,1].set_xticks([])            
        fig.align_labels()
        fig.savefig("./figures/errors_skill_%s.png"%skill, format = "png", dpi=dpi)            
        print group_errors_by_csv.keys()
        np.save('group_errors_by_csv_skill%s.npy'%skill,group_errors_by_csv)
        np.save('group_stamps_by_csv_skill%s.npy'%skill, group_stamps_by_csv)    
        import plot_errors_for_illustrating_in_diff_and_same_lengths
        plot_errors_for_illustrating_in_diff_and_same_lengths.run(group_errors_by_csv = group_errors_by_csv,
                                                              group_stamps_by_csv = group_stamps_by_csv,
                                                              c = 3.0)
#-----------------------Figure-2----------------        
        # plot durations distribution
        fig, axarr = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(14,6))
        plt.subplots_adjust(top=0.91, hspace=0.8,bottom=0.1)
        plt.axes(axarr[0])        
        count, bins, ignored = plt.hist(durations, 30, density=True)
        mu, sigma = np.mean(np.array(durations)), np.std(np.array(durations))
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
        plt.title("Normal distribution of validation movements")
        plt.xlabel("Duration (s)")
        plt.ylabel("Probability")        

    list_of_c   = np.linspace(0.5, 5, 10).tolist()[::-1] # [3]: explaining the results
    list_of_tpr = []
    list_of_fpr = []
    for i, c in enumerate(list_of_c):
        tpr, fpr, list_of_stamps, list_of_estimated_alpha, list_of_csv_type = testing_results(
                        succ_csvs = valid_succ_csvs,
                        unsucc_csvs = unsucc_csvs,
                        n_lags = n_lags,
                        prediction_model = model,
                        ipromp_model = ipromp_model,
                        c = c,
                        skill = skill)
        print ("tpr:%s, fpr:%s"%(tpr, fpr))
        list_of_tpr.append(tpr)
        list_of_fpr.append(fpr)
        if GENERATE_FIGURE_IN_PAPER:        
            if i == 0: # only for plot
                plt.axes(axarr[1])        
                plt.title('Estimated alpha values of testing movements')
                for j, csv_type in enumerate(list_of_csv_type):
                    color = "blue" if csv_type == "succ" else "red"
                    plt.scatter(list_of_stamps[j], list_of_estimated_alpha[j], c = color, label=csv_type)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.xlabel('Duration (s)') 
                plt.ylabel('Estimated Alpha (s)')   
                plt.savefig('./figures/stamp_vs_alpha_skill%s.png'%skill, format="png", dpi=dpi)

    np.save('list_of_tpr_skill%s.npy'%skill,list_of_tpr)
    np.save('list_of_fpr_skill%s.npy'%skill,list_of_fpr)
    plt.figure()
    plt.plot(list_of_fpr, list_of_tpr, lw=2.0, label="ROC of skill%s"%skill)
    plt.legend()
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')   
    plt.savefig('./figures/roc_skill%s.png'%skill, format="png", dpi=dpi)
    
