
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.regularizers import l1, activity_l1, l2, activity_l2,l1l2
from keras.wrappers.scikit_learn import KerasRegressor

import xgboost as xgb


def create_GLM_model(l1,l2, optimizer, input_dim=196):

    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, init='uniform', activation='linear',W_regularizer=l1l2(l1=l1, l2=l2)))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer=optimizer)

    return model

def create_1l_nn_model(l1,l2,act,n_neurons, dropout, optimizer, input_dim=196):

    model = Sequential()
    model.add(Dense(n_neurons, input_dim=input_dim, init='uniform', activation=act, W_regularizer=l1l2(l1=l1, l2=l2)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer=optimizer)

    return model

def create_2l_nn_model(l1, l2, n_neurons_1l, n_neurons_2l, dropout1, dropout2, act1, act2, optimizer, input_dim=196):

    model = Sequential()
    model.add(Dense(n_neurons_1l, input_dim=input_dim, init='uniform', activation=act1, W_regularizer=l1l2(l1=l1, l2=l2)))
    model.add(Dropout(dropout1))
    model.add(Dense(n_neurons_2l, init='uniform', activation=act2, W_regularizer=l1l2(l1=l1, l2=l2)))
    model.add(Dropout(dropout2))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer=optimizer)

    return model


def create_rnn_model(nLSTM, dropout, act, l1, l2, optimizer, input_dim=196,input_length=5):

    model = Sequential()
    model.add(LSTM(nLSTM, input_dim=input_dim,input_length=input_length, activation=act, W_regularizer=l1l2(l1=l1, l2=l2)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer=optimizer)

    return model