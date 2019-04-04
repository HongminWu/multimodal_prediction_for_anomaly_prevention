from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
from pandas import DataFrame
from pandas import concat

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_model_via_name(model_name=None, input_shape=None, output_shape=None):
    if model_name == "lstm":
        # design lstm network
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(output_shape))
        model.compile(loss='mae', optimizer='adam')
        
    elif model_name == 'lstm_dropout':
        # design lstm network
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))        
        model.add(Dense(output_shape))
        model.compile(loss='mae', optimizer='adam')

    elif model_name == 'gru_dropout':
        # design lstm network
        model = Sequential()
        model.add(GRU(10, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(10))
        model.add(Dropout(0.2))        
        model.add(Dense(output_shape))
        model.compile(loss='mae', optimizer='adam')
        
    else:
        pass
    return model