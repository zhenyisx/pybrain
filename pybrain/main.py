#!/usr/bin/python


# -*- coding: utf-8 -*-

# from __future__ import print_function
import pandas as pd
import os
from os import walk
import re
import numpy as np
import cPickle as pkl
import sys
from operator import add
import operator
import csv

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
# from keras.utils.visualize_util import plot
from keras.layers.embeddings import Embedding
# from keras.layers import containers
from keras.utils import np_utils
from keras.models import model_from_json
from keras.regularizers import ActivityRegularizer,l1,activity_l1

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle

from dataset import load_full_data
from dataset import load_over_data



from config import data_info

np.random.seed(1337)  # for reproducibility





def demo_seq2seq_saelstm(n_timesteps=6):
    """Sequence to sequence of labels classification using stacked AutoEncoders
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """

    # model parameters
    pretrain_batch_size = 20
    finetune_batch_size = 20
    pretrain_nb_epochs = 100
    finetune_nb_epochs = 200
    pretrain_nb_hidden_layers = [630, 256, 128, 64]
    finetune_nb_hidden_units = 32

    # load data
    X, y, kf = load_full_data(name_pkl='seq_mci_shuf_cv.p')

    # model training and testing
    it = 0
    score_list = []
    for train_index, test_index in kf:
        X_train, X_test = \
                X[train_index, :n_timesteps, :], X[test_index, :n_timesteps, :]
        y_train, y_test = \
                y[train_index, :n_timesteps], y[test_index, :n_timesteps]

        X_train = X_train.reshape(X_train.shape[0], -1, 630)
        X_test = X_test.reshape(X_test.shape[0], -1, 630)
        y_train = y_train.reshape(y_train.shape[0], -1, 1)
        y_test = y_test.reshape(y_test.shape[0], -1, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        it += 1
        print('Fold %d Training and Evaluation...' % it)
        trained_encoders = []
        X_train_tmp = X_train  # input shape (n_samples, n_steps, n_dim)
        for n_in, n_out in zip(pretrain_nb_hidden_layers[:-1],
                               pretrain_nb_hidden_layers[1:]):
            print('Pre-training the layer: Input {} -> Output {}'.format(n_in,
                                                                         n_out))
            # Create AE and training
            ae = Sequential()
            ae.add(AutoEncoder(encoder=LSTM(n_out,
                                            input_dim=n_in,
                                            return_sequences=True),
                               decoder=LSTM(n_in,
                                            input_dim=n_out,
                                            return_sequences=True),
                               output_reconstruction=False)
                  )  # output shape: (nb_samples, timesteps, 10)
            optimizer = RMSprop(lr=0.001, clipnorm=10)
            ae.compile(optimizer=optimizer, loss='mse')
            ae.fit(
                X_train_tmp,
                X_train_tmp,
                batch_size=pretrain_batch_size,
                nb_epoch=pretrain_nb_epochs)  # batch_size=pretrain_batch_size,
            # Store trainined weight
            trained_encoders.append(ae.layers[0].encoder)
            # Update training data
            X_train_tmp = ae.predict(X_train_tmp)

        # Fine-tuning
        print('Fine-tuning')
        for encoder in trained_encoders:
            model.add(encoder)
        model.add(Dropout(0.5))
        model.add(TimeDistributedDense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      class_mode='binary')
        model.fit(X_train,
                  y_train,
                  nb_epoch=finetune_nb_epochs,
                  batch_size=finetune_batch_size,
                  show_accuracy=True,
                  verbose=1,
                  validation_split=0.1)
        y_predict = model.predict_classes(X_test)
        scores = [accuracy_score(
            np.reshape(y_test, (-1, 1)), np.reshape(y_predict, (-1, 1))),
                  f1_score(
                      np.reshape(y_test, (-1, 1)), np.reshape(y_predict,
                                                              (-1, 1)))]
        score_list.append(scores)
        print scores

    # results presentation
    print(model.get_config())
    for s in score_list:
        print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
    print('Mean Accuracy: %.4f\tMean F1: %.4f' %
          (np.mean([s[0] for s in score_list]),
           np.mean([s[1] for s in score_list])))


def demo_seq2label_saelstm(n_timesteps=6):
    """Sequence to label classification using stacked AutoEncoders + LSTM
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """

    # model parameters
    # pretrain_batch_size = 20
    # finetune_batch_size = 20
    pretrain_nb_epochs = 100
    finetune_nb_epochs = 200
    pretrain_nb_hidden_layers = [630, 256, 128, 64]
    finetune_nb_hidden_units = 32

    # load data
    X, y, kf = load_full_data()

    # model evaluation
    it = 0
    score_list = []
    for train_index, test_index in kf:
        X_train, X_test = \
                X[train_index, :n_timesteps, :], X[test_index, :n_timesteps, :]
        y_train, y_test = y[train_index], y[test_index]

        # X_train = X_train.reshape(X_train.shape[0], -1, 630)
        # X_test = X_test.reshape(X_test.shape[0], -1, 630)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        it += 1
        print('Fold %d Training and Evaluation...' % it)
        # Layer-wise pre-training
        trained_encoders = []
        X_train_tmp = X_train  # input shape (n_samples, n_steps, n_dim)
        for n_in, n_out in zip(pretrain_nb_hidden_layers[:-1],
                               pretrain_nb_hidden_layers[1:]):
            print('Pre-training the layer: Input {} -> Output {}'.format(n_in,
                                                                         n_out))
            # Create AE and training
            ae = Sequential()
            # ae.add(TimeDistributedDense(n_in,input_dim=n_in)) # output shape: (nb_samples, timesteps, 10)
            ae.add(AutoEncoder(encoder=LSTM(n_out,
                                            input_dim=n_in,
                                            return_sequences=True),
                               decoder=LSTM(n_in,
                                            input_dim=n_out,
                                            return_sequences=True),
                               output_reconstruction=False)
                  )  # output shape: (nb_samples, timesteps, 10)
            optimizer = RMSprop(lr=0.001, clipnorm=10)
            ae.compile(optimizer=optimizer, loss='mse')
            ae.fit(
                X_train_tmp,
                X_train_tmp,
                nb_epoch=pretrain_nb_epochs)  # batch_size=pretrain_batch_size,
            # Store trainined weight
            trained_encoders.append(ae.layers[0].encoder)
            # Update training data
            X_train_tmp = ae.predict(X_train_tmp)
        # Fine-tuning
        print('Fine-tuning')
        model = Sequential()
        for encoder in trained_encoders:
            model.add(encoder)
        model.add(LSTM(finetune_nb_hidden_units,
                       activation='sigmoid',
                       inner_activation='hard_sigmoid',
                       input_dim=pretrain_nb_hidden_layers[-1],
                       return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      class_mode='binary')
        model.fit(X_train,
                  y_train,
                  nb_epoch=finetune_nb_epochs,
                  show_accuracy=True,
                  verbose=1,
                  validation_split=0.1)
        y_predict = model.predict_classes(X_test)
        scores = [accuracy_score(y_test, y_predict),
                  f1_score(y_test, y_predict)]
        score_list.append(scores)
        print scores

    # results presentation
    print(model.get_config())
    for s in score_list:
        print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
    print('Mean Accuracy: %.4f\tMean F1: %.4f' %
          (np.mean([s[0] for s in score_list]),
           np.mean([s[1] for s in score_list])))


def demo_seq2seq_lstm(n_timesteps=6):
    """Sequence to sequence (label sequence) classification using LSTM
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """

    # model parameters
    # batch_size = 20
    # nb_classes = 10
    nb_epochs = 200
    hidden_units = 64

    # load data
    X, y, kf = load_full_data(name_pkl='seq_mci_shuf_cv.p')

    # model structure
    model = Sequential()
    model.add(LSTM(hidden_units,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=X.shape[-1],
                   return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributedDense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode='binary')

    # model training and testing
    it = 0
    score_list = []
    for train_index, test_index in kf:
        X_train, X_test = \
                X[train_index, :n_timesteps, :], X[test_index, :n_timesteps, :]
        y_train, y_test = \
                y[train_index, :n_timesteps], y[test_index, :n_timesteps]

        X_train = X_train.reshape(X_train.shape[0], -1, 630)
        X_test = X_test.reshape(X_test.shape[0], -1, 630)
        y_train = y_train.reshape(y_train.shape[0], -1, 1)
        y_test = y_test.reshape(y_test.shape[0], -1, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        it += 1
        print('Fold %d Training and Evaluation...' % it)
        model.fit(X_train,
                  y_train,
                  nb_epoch=nb_epochs,
                  show_accuracy=True,
                  verbose=1,
                  validation_split=0.1)
        y_predict = model.predict_classes(X_test)
        scores = [accuracy_score(
            np.reshape(y_test, (-1, 1)), np.reshape(y_predict, (-1, 1))),
                  f1_score(
                      np.reshape(y_test, (-1, 1)), np.reshape(y_predict,
                                                              (-1, 1)))]
        score_list.append(scores)
        print scores

    # results presentation
    print(model.get_config())
    for s in score_list:
        print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
    print('Mean Accuracy: %.4f\tMean F1: %.4f' %
          (np.mean([s[0] for s in score_list]),
           np.mean([s[1] for s in score_list])))


def demo_seq2label_lstm(n_timesteps=6):
    """Sequence to label classification using LSTM
        Input:
            n_timesteps: the length of sequences (# of time steps)
        Output: 
            label: 1 (converted) or 0 (not converted)
    """

    # model paramters
    # batch_size = 20
    # nb_classes = 10
    nb_epochs = 200
    hidden_units = 64
    # mpath = '/home/zhen/Projects/Data/MCI/'
    # load data
    # X, y, kf = load_full_data(
    #     name_pkl='mci2ad_%dt_prediction_shuf_cv.p' % n_timesteps)
    X, y, kf = load_over_data(
        name_pkl='mci2ad_%dt_prediction_over_shuf_cv.p' % n_timesteps)

    # model structure
    model = Sequential()
    model.add(LSTM(hidden_units,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=X.shape[-1],
                   input_length=n_timesteps,
                   return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1,W_regularizer=l1(0.01), activity_regularizer = activity_l1(0.01)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode='binary')

    # model training and testing
    it = 0
    score_list = []
    for train_index, test_index in kf:
        X_train, X_test = \
                X[train_index, :n_timesteps, :], X[test_index, :n_timesteps, :]
        y_train, y_test = y[train_index], y[test_index]
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        it += 1
        print('Fold %d Training and Evaluation...' % it)
        model.fit(X_train,
                  y_train,
                  nb_epoch=nb_epochs,
                  show_accuracy=True,
                  verbose=1,
                  validation_split=0.1)
        y_predict = model.predict_classes(X_test)
        # print y_test
        # print y_predict
        # raw_input('wait')
        scores = [accuracy_score(y_test, y_predict),
                  f1_score(y_test, y_predict)]
        score_list.append(scores)
        print scores
        json_string = model.to_json()
#         open(data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_tf.json' %
#              (n_timesteps, it), 'w').write(json_string)
#         model.save_weights(
#             data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_weights_tf.h5' % (
#                 n_timesteps, it))

    # results presentation
    print(model.get_config())
    for s in score_list:
        print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
    print('Mean Accuracy: %.4f\tMean F1: %.4f' %
          (np.mean([s[0] for s in score_list]),
           np.mean([s[1] for s in score_list])))


 

def demo_seq2label_lstm_overdata(n_timesteps=6):
    """Sequence to label classification using LSTM
        Description: using oversampled data
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """
 
    # model paramters
    # batch_size = 20
    # nb_classes = 10

    nb_epochs = 200
    hidden_units = 64
    ndim=630
    dataset = load_over_data(
        name_pkl='mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps)
 
    # model structure
    model = Sequential()
    model.add(LSTM(hidden_units,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=ndim,
                   input_length=n_timesteps,
                   return_sequences=False,
                   W_regularizer=l1(0.00001),
                   U_regularizer=l1(0.00001),
                   b_regularizer=l1(0.00001)))
    model.add(Dropout(0.5))
    # model.add(Dense(1,W_regularizer=l1(0.05), activity_regularizer = activity_l1(0.05)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode='binary')
 
    # model training and testing
    score_list = []
    for it in range(len(dataset)):
        X_train, y_train, X_test, y_test = \
                dataset[it][0],dataset[it][1], dataset[it][2], dataset[it][3]
        X_train = np.asarray(X_train,dtype='float32')
        X_test = np.asarray(X_test,dtype='float32')
        y_train = np.asarray(y_train,dtype='int32')
        y_test = np.asarray(y_test,dtype='int32')
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        print('Fold %d Training and Evaluation...' % (it+1))
        model.fit(X_train,
                  y_train,
                  nb_epoch=nb_epochs,
                  show_accuracy=True,
                  verbose=1,
                  validation_split=0.1)
        y_predict = model.predict_classes(X_test)
        scores = [accuracy_score(y_test, y_predict),
                  f1_score(y_test, y_predict)]
        score_list.append(scores)
        print scores
        json_string = model.to_json()
        open(data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' %
             (n_timesteps, it), 'w').write(json_string)
        model.save_weights(
            data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' % (
                n_timesteps, it))

    # results presentation
    print(model.get_config())
    for s in score_list:
        print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
    print('Mean Accuracy: %.4f\tMean F1: %.4f' %
          (np.mean([s[0] for s in score_list]),
           np.mean([s[1] for s in score_list])))
#         # y_predict = model.predict_classes(X_test)
#         # print y_test
#         # print y_predict
#         # raw_input('wait')
#         # score_list.append(model.evaluate(X_test, y_test))
#         score_list.append(np.sqrt(model.evaluate(X_test, y_test)))
#         print score_list
#         # json_string = model.to_json()
#         # open(mpath + 'mci2ad_%dt_regression_cv%d_model_tf.json' %
#         #      (n_timesteps, it), 'w').write(json_string)
#         # model.save_weights(mpath +
#         #                    'mci2ad_%dt_regression_cv%d_model_weights_tf.h5' % (
#         #                        n_timesteps, it))
#  
#         # results presentation
#     print(model.get_config())
#     for s in score_list:
#         print('Error: %.4f' % (s))
#     print('Patients: %d \t Time Steps: %d \t Mean Error: %.4f' %
#           (len(y_test), n_timesteps, np.mean(score_list)))
#     print('score statistics: max %.4f, min %.4f, mean %.4f, std %.4f' %
#           (np.max(y_test), np.min(y_test), np.mean(y_test), np.std(y_test)))
# 
# 
# def demo_seq2score_linear(n_timesteps=6):
#     """Sequence to score regression using linear regression
#         Input:
#             n_timesteps: the length of sequences (# of time steps)
#     """
# 
#     # model paramters
#     # batch_size = 20
#     # nb_classes = 10
#     nb_epochs = 300
#     hidden_units = 128
#     mpath = '/home/zhen/Projects/Data/MCI/'
#     # load data
#     # X, y, kf = load_full_data(
#     #     name_pkl='mci2ad_%dt_prediction_shuf_cv.p' % n_timesteps)
#     X, y, kf = load_full_data(
#         name_pkl='mci2ad_%dt_regression_dif_shuf_cv.p' % n_timesteps)
#     print('score statistics: max %.4f, min %.4f, mean %.4f, std %.4f' %
#           (np.max(y), np.min(y), np.mean(y), np.std(y)))
#     y = y - np.mean(y)
#     # print type(X),type(y)
#     # raw_input('wait')
# 
#     regr = linear_model.LinearRegression()
# 
#     # model training and testing
#     it = 0
#     score_list = []
#     for train_index, test_index in kf:
#         X_train, X_test = \
#                 X[train_index, :n_timesteps, :], X[test_index, :n_timesteps, :]
# 
#         # X_train, X_test = X[train_index, :, :], X[test_index, :, :]
#         y_train, y_test = y[train_index], y[test_index]
#         # print type(X_train),type(y_train)
#         # print X_train.shape, y_train.shape
#         # raw_input('wait')
#         # X_train = X_train.reshape(X_train.shape[0], -1).astype = ('float32')
#         # X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
#         X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float32')
#         X_test = np.reshape(X_test,(X_test.shape[0], -1)).astype('float32')
#         it += 1
#         print('Fold %d Training and Evaluation...' % it)
#         # Train the model using the training sets
#         regr.fit(X_train, y_train)
#         y_pred = regr.predict(X_test)
#         # The coefficients
#         # print('Coefficients: \n', regr.coef_)
#         # The mean square error
#         # print("Residual sum of squares: %.2f"
#         #             % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) **
#         #                       2))
#         # Explained variance score: 1 is perfect prediction
#         # print('Variance score: %.2f' % regr.score(diabetes_X_test,
#         #                                       diabetes_y_test))
#         score_list.append((np.sqrt(mean_squared_error(y_test, y_pred)),
#                            mean_absolute_error(y_test, y_pred)))
#         print score_list
# 
#     # results presentation
#     # print(model.get_config())
#     for s in score_list:
#         print('MAE: %.4f\t RMSE: %.4f' % (s[1], s[0]))
#     print('Patients: %d \t Time Steps: %d \t MSE: %.4f \t RMSE:%.4f' %
#           (len(y), n_timesteps, np.mean([s[1] for s in score_list]),
#            np.mean([s[0] for s in score_list])))
#     print('score statistics: max %.4f, min %.4f, mean %.4f, std %.4f' %
#           (np.max(y), np.min(y), np.mean(y), np.std(y)))
# =======
#                   verbose=2,
#                   validation_split=0.1)
#         y_predict = model.predict_classes(X_test)
#         # print y_test
#         # print y_predict
#         # raw_input('wait')
#         scores = [accuracy_score(y_test, y_predict),
#                   f1_score(y_test, y_predict)]
#         score_list.append(scores)
#         print scores
#         json_string = model.to_json()
#         open(mpath + 'mci2ad_%dt_prediction_over2_cv%d_model_tf.json' %
#              (n_timesteps, it), 'w').write(json_string)
#         model.save_weights(
#             mpath + 'mci2ad_%dt_prediction_over2_cv%d_model_weights_tf.h5' % (
#                 n_timesteps, it))
# 
#     # results presentation
#     print(model.get_config())
#     for s in score_list:
#         print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
#     print('Mean Accuracy: %.4f\tMean F1: %.4f' %
#           (np.mean([s[0] for s in score_list]),
#            np.mean([s[1] for s in score_list])))
# >>>>>>> 1ace2cd59cafbd457b85d782ce5b37c5f41a23ec
# 
# 
def demo_model_analysis_intersection(n_timesteps=6):
    """Sequence to label classification using LSTM
        Input:
            n_timesteps: the length of sequences (# of time steps)
        Return the list of regions that appear most frequently
    """
 
    region_list = range(90)
    topK = 20
    print n_timesteps, topK
    for it in range(5):
        print('Fold %d Statistics...' % (it + 1))
        model = model_from_json(open(
            data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' % (
                n_timesteps, it)).read())
        model.load_weights(
            data_info['path'] + 'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' % (
                n_timesteps, it))
        for layer in model.layers:
            if layer.get_config()['name'] == 'LSTM':
                g = layer.get_config()
                h = layer.get_weights()
                h1 = np.mean(h[0::3], axis=0)
                score = __region_rank(h1)
                # print score
                score = np.asarray(score)
                si = np.argsort(score, axis=0)[::-1]
                # print list(si[:topK])
                # print region_list
                region_list = list(set(region_list) & set(list(si[:topK])))
    print region_list
 
 
def demo_model_analysis(n_timesteps=6):
    """Evaluate region weights
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """
    n_region = 90
#     mpath = '/home/zhen/Projects/Data/MCI/'
    # region_list = range(90)
    region_score = [0] * n_region
    topK = 20
    print n_timesteps, topK
    for it in range(5):
        print('Fold %d Statistics...' % (it + 1))
        try:
            model = model_from_json(open(
                data_info['path']  + 'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' % (
                    n_timesteps, it )).read())
            model.load_weights(
                data_info['path']  + 'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' % (
                    n_timesteps, it ))
            for layer in model.layers:
    #             print layer.get_config()
    #             raw_input('wait')
                if layer.get_config()['name'] == 'lstm_1':
                    g = layer.get_config()
                    h = layer.get_weights()
    #                 print h
    #                 raw_input('wait')
                    h1 = np.mean(h[0::3],
                                 axis=0)  # three matrices at each time step
                    score = _region_rank(h1)
                    region_score = map(add, score, region_score)  # add values in two lists
        except IOError:
            pass

    region_score = np.asarray(region_score)  # score for all regions
    sumofscore = np.sum(region_score)
#     print region_score
#     print sumofscore
#     raw_input('wait')
    dfile = 'RegionMap90a.txt'
    dict_region = dict(csv.reader(open(data_info['path']  + dfile, 'rb'), delimiter=' '))
    region_scoredict = dict()
    for region_id in range(len(region_score)):
        if dict_region[str(region_id)] in region_scoredict:
            region_scoredict[dict_region[str(region_id)]] += float(region_score[
                region_id]) / float(sumofscore)
        else:
            region_scoredict[dict_region[str(region_id)]] = float(region_score[
                region_id]) / float(sumofscore)
    sorted_region_scoredict = sorted(region_scoredict.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)
    # print sorted_region_scoredict
    for i in sorted_region_scoredict:
        print i[0] + '\t' + str(i[1])
    # si = np.argsort(region_score, axis=0)[::-1]
    # print si
    # print list(si[:topK])
    # print region_list
    # region_list =  list(set(region_list)&set(list(si[:topK])))
    # print ' '.join(['%d:%.3f'% (idx,region_score[idx]) for idx in si])
    # si[:]
 
 
def _region_rank(h):
    """Return list of region importance values
        * h is a weight matrix
    """
    n_region = 90
    n_rdim = 7
    v = np.mean(np.absolute(h), axis=1)  # n_rdim*n_region long vector
    # print v.shape
    rscore = [0] * n_region
    index_base_list = [0, 90, 180, 270, 271, 272, 273]
    for s in range(n_region):
        index_list = [s + x for x in index_base_list]
        # print index_list
        rscore[s] += np.sum([v[x] for x in index_list])
    return rscore
 
 
# def demo_sklearn_crossvalidation(model_type):
#     """k-fold cross validation for logistic regression/lstm
#     """
#     # batch_size = 20
#     # nb_classes = 10
#     nb_epochs = 500
#     hidden_units = 64
# 
#     # learning_rate = 1e-6
#     # clip_norm = 1.0
# 
#     # the data, shuffled and split between train and test sets
#     X, y, kf = load_full_data()
#     # pdb.set_trace()
#     # print(X.shape)
#     # print(y.shape)
#     # print(y)
# 
#     # compile model
#     if model_type == 'lr':
#         print('Logistic Regression Model')
#         model = linear_model.LogisticRegression(C=1e5)
#     elif model_type == 'lstm':
#         print('LSTM Model')
#         model = Sequential()
#         print(X.shape[-1])
#         # model.add(Embedding(X.shape[-1], 128 ))
#         model.add(LSTM(hidden_units,
#                        activation='sigmoid',
#                        inner_activation='hard_sigmoid',
#                        input_dim=X.shape[-1],
#                        return_sequences=False))
#         # model.add(LSTM(hidden_units,
#         #                input_shape=X.reshape(X.shape[0], -1, 1).shape[1:]))
#         model.add(Dropout(0.5))
#         model.add(Dense(1))
#         model.add(Activation('sigmoid'))
#         # rmsprop = RMSprop(lr=learning_rate)
#         # raw_input('wait')
#         # model.compile(loss='binary_crossentropy', optimizer=rmsprop)
#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       class_mode='binary')
#     # plot(model, to_file='mci_lstm_model.png')
#     print('model compiled successfully.')
# 
#     it = 0
#     score_list = []
#     for train_index, test_index in kf:
#         # print(test_index)
#         X_train, X_test = X[train_index, :, :], X[test_index, :, :]
#         y_train, y_test = y[train_index], y[test_index]
# 
#         # print('X_train shape:', X_train.shape)
#         # print('y_train shape:', y_train.shape)
#         # X_train = X_train.reshape(X_train.shape[0], -1, 630)
#         # X_test = X_test.reshape(X_test.shape[0], -1, 630)
#         X_train = X_train.astype('float32')
#         X_test = X_test.astype('float32')
#         # print('X_train shape:', X_train.shape)
#         # print('y_train shape:', y_train.shape)
#         # print(X_train.shape[0], 'train samples')
#         # print(X_test.shape[0], 'test samples')
#         # raw_input('wait')
#         it += 1
#         print('Fold %d Evaluation...' % it)
#         # model.fit(X_train, y_train)
#         model.fit(X_train,
#                   y_train,
#                   nb_epoch=nb_epochs,
#                   show_accuracy=True,
#                   verbose=1,
#                   validation_split=0.1)
#         # plot(model, to_file='rnn_model.png')
#         # json_string = model.to_json()
#         # open('rnn_architecture.json', 'w').write(json_string)
#         # model.save_weights('rnn_weights.h5')
#         y_predict = model.predict_classes(X_test)
#         # print(y_test)
#         # print(y_predict)
#         scores = [accuracy_score(y_test, y_predict),
#                   f1_score(y_test, y_predict)]
#         # scores = model.evaluate(X_test, y_test, 
#         #                         show_accuracy=True, 
#         #                         verbose=0)
#         # print('test score: %.4f test accuracy: %.4f' %
#         # (scores[0], scores[1]))
#         score_list.append(scores)
#     for s in score_list:
#         print('Acc: %.4f\tF1: %.4f' % (s[0], s[1]))
#     print('LSTM size: %d\tMean Accuracy: %.4f\tMean F1: %.4f' %
#           (hidden_units, np.mean([s[0] for s in score_list]),
#            np.mean([s[1] for s in score_list])))
# 
# 
# def demo_crossvalidation(model_type):
#     """k-fold cross validation for simpleRNN/lstm
#     """
# 
#     batch_size = 10
#     # nb_classes = 10
#     nb_epochs = 5
#     hidden_units = 100
# 
#     # learning_rate = 1e-6
#     # clip_norm = 1.0
# 
#     # the data, shuffled and split between train and test sets
#     X, y, kf = load_full_data()
#     # print(X.shape)
#     # print(y.shape)
# 
#     # compile model
#     if model_type == 'irnn':
#         print('IRNN Model')
#         model = Sequential()
#         model.add(SimpleRNN(output_dim=hidden_units,
#                             init=lambda shape: normal(shape, scale=0.001),
#                             inner_init=lambda shape: identity(shape, scale=1.0),
#                             activation='relu',
#                             input_shape=X.reshape(X.shape[0], -1, 1).shape[1:]))
#         model.add(Dense(1))
#         model.add(Activation('sigmoid'))
#         # rmsprop = RMSprop(lr=learning_rate)
#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       class_mode='binary')
#     elif model_type == 'lstm':
#         print('LSTM Model')
#         model = Sequential()
#         model.add(LSTM(hidden_units,
#                        input_shape=X.reshape(X.shape[0], -1, 1).shape[1:]))
#         model.add(Dense(1))
#         model.add(Activation('sigmoid'))
#         # rmsprop = RMSprop(lr=learning_rate)
#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       class_mode='binary')
# 
#     it = 0
#     score_list = []
#     for train_index, test_index in kf:
#         X_train, X_test = X[train_index, :, :], X[test_index, :, :]
#         y_train, y_test = y[train_index], y[test_index]
#         X_train = X_train.reshape(X_train.shape[0], -1, 1)
#         X_test = X_test.reshape(X_test.shape[0], -1, 1)
#         X_train = X_train.astype('float32')
#         X_test = X_test.astype('float32')
#         # print('X_train shape:', X_train.shape)
#         # print(X_train.shape[0], 'train samples')
#         # print(X_test.shape[0], 'test samples')
#         # raw_input('wait')
#         it += 1
#         print('Fold %d Evaluation...' % it)
#         model.fit(X_train,
#                   y_train,
#                   batch_size=batch_size,
#                   nb_epoch=nb_epochs,
#                   show_accuracy=True,
#                   verbose=1,
#                   validation_split=0.1)
#         # plot(model, to_file='rnn_model.png')
#         # json_string = model.to_json()
#         # open('rnn_architecture.json', 'w').write(json_string)
#         # model.save_weights('rnn_weights.h5')
#         scores = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
#         # y_predp = model.predict(X_test, verbose=0)
#         y_predc = model.predict_classes(X_test, verbose=0)
#         # print( y_predp )
#         # print( y_predc )
#         # print( y_test )
#         # raw_input('wait')
#         scores.append(accuracy_score(y_test, y_predc))
#         # print('test score: %.4f test accuracy: %.4f' % (scores[0], scores[1]))
#         score_list.append(scores)
#     for s in score_list:
#         print('test score: %.4f test accuracy: %.4f accuracy: %.4f' %
#               (s[0], s[1], s[2]))
# 
# 
# def demo_compare_model():
#     '''compare simple RNN and LSTM'''
#     batch_size = 10
#     # nb_classes = 10
#     nb_epochs = 5
#     hidden_units = 10
# 
#     learning_rate = 1e-6
#     # clip_norm = 1.0
# 
#     # the data, shuffled and split between train and test sets
#     X_train, y_train, X_test, y_test = mci.load_data()
# 
#     print(X_train.shape)
# 
#     X_train = X_train.reshape(X_train.shape[0], -1, 1)
#     X_test = X_test.reshape(X_test.shape[0], -1, 1)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     print('X_train shape:', X_train.shape)
#     print(X_train.shape[0], 'train samples')
#     print(X_test.shape[0], 'test samples')
# 
#     # convert class vectors to binary class matrices
#     # Y_train = np_utils.to_categorical(y_train, nb_classes)
#     # Y_test = np_utils.to_categorical(y_test, nb_classes)
# 
#     print('Evaluate IRNN...')
#     model = Sequential()
#     model.add(SimpleRNN(output_dim=hidden_units,
#                         init=lambda shape: normal(shape, scale=0.001),
#                         inner_init=lambda shape: identity(shape, scale=1.0),
#                         activation='relu', input_shape=X_train.shape[1:]))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     rmsprop = RMSprop(lr=learning_rate)
#     model.compile(loss='binary_crossentropy', optimizer=rmsprop)
# 
#     model.fit(X_train,
#               y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epochs,
#               show_accuracy=True,
#               verbose=0,
#               validation_split=0.1)
#     plot(model, to_file='rnn_model.png')
#     json_string = model.to_json()
#     open('rnn_architecture.json', 'w').write(json_string)
#     model.save_weights('rnn_weights.h5')
#     scores = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
#     print('IRNN test score:', scores[0])
#     print('IRNN test accuracy:', scores[1])
# 
#     print('Compare to LSTM...')
#     model = Sequential()
#     model.add(LSTM(hidden_units, input_shape=X_train.shape[1:]))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     rmsprop = RMSprop(lr=learning_rate)
#     model.compile(loss='binary_crossentropy', optimizer=rmsprop)
# 
#     model.fit(X_train,
#               y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epochs,
#               show_accuracy=True,
#               verbose=0,
#               validation_split=0.1)
#     plot(model, to_file='lstm_model.png')
#     json_string = model.to_json()
#     open('lstm_architecture.json', 'w').write(json_string)
#     model.save_weights('lstm_weights.h5')
#     scores = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
#     print('LSTM test score:', scores[0])
#     print('LSTM test accuracy:', scores[1])
# 
# 


def main():
    """Main entry of the demo
    """
    print('Welcome to our demo!')
    print(
        'Please input method type: 1:seq2label 2:seq2seq 3:seq2label+sae 4:seq2seq+sae 5:analysis 6:seq2score'
    )
    demoid = int(raw_input())
    if demoid == 1:
        # demo_seq2label_lstm(n_timesteps=1)
        demo_seq2label_lstm_overdata(n_timesteps=5)
    elif demoid == 2:
        demo_seq2seq_lstm(n_timesteps=3)
    elif demoid == 3:
        demo_seq2label_saelstm()
    elif demoid == 4:
        demo_seq2seq_saelstm(n_timesteps=5)
    elif demoid == 5:
        demo_model_analysis(n_timesteps=5)
#         demo_model_analysis_intersection(n_timesteps=1)
    elif demoid == 6:
        # demo_seq2score_lstm(n_timesteps=int(raw_input('Please input # of time steps:')))
        demo_seq2score_linear(
            n_timesteps=int(raw_input('Please input # of time steps:')))
    else:
        print('demo type not supported')


if __name__ == '__main__':
    main()

