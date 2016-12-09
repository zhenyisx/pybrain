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
from operator import add, le
import operator
import csv
import time

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
from keras.regularizers import ActivityRegularizer, l1, activity_l1
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle

from dataset import load_full_data
from dataset import load_over_data

from config import data_info

np.random.seed(1337)  # for reproducibility


def power_estimation(n_timesteps=6, n_stages=0):
    """
    power estimation for rnn models
     -- permute the labels
     -- train rnn with permuted labels
     -- compute the p-value
    """
    n_permutation = 1000
    nb_epochs = 200
    hidden_units = 64
    ndim = 630
    if n_stages == 0:
        dataset = load_over_data(
            name_pkl='mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps)
    else:
        dataset = load_over_data(
            name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
            (n_timesteps, n_stages))

    # model structure
    model = Sequential()
    model.add(
        LSTM(
            hidden_units,
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
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')

    # model training and testing
    #
    pvals = []
    for it in range(0, min(1, len(dataset))):
        X_train, y_train, X_test, y_test = \
                dataset[it][0],dataset[it][1], dataset[it][2], dataset[it][3]
        X_train = np.asarray(X_train, dtype='float32')
        X_test = np.asarray(X_test, dtype='float32')
        y_test = np.asarray(y_test, dtype='int32')
        y_train = np.asarray(y_train, dtype='int32')
        y_train_list = [y_train]
        for it2 in range(n_permutation):
            y_train_list.append(np.random.permutation(y_train))
            # print y_train_list
            # raw_input()
        score_list = []
        for y_train in y_train_list:
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            print('Fold %d Training and Evaluation...' % (it + 1))
            model.fit(X_train,
                      y_train,
                      nb_epoch=nb_epochs,
                      show_accuracy=True,
                      verbose=1,
                      validation_split=0.1)
            y_predict = model.predict_classes(X_test)
            y_predict = [l[0] for l in y_predict]
            if sum(y_predict) == 0 or sum(y_predict) == len(y_predict):
                continue
            fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
            cid = list(thresholds).index(1)
            scores = [accuracy_score(y_test, y_predict),
                      f1_score(y_test, y_predict), tpr[cid], 1 - fpr[cid]]
            score_list.append(scores)
        # evaluate p-value
        # print score_list
        pval_list = [0] * len(score_list[0])
        for i in range(1, len(score_list)):
            pval_list = map(add, pval_list, map(float, map(le, score_list[0],
                                                           score_list[i])))
        #     print pval_list
        # print pval_list
        pval_list = map(lambda x: x / float(len(score_list) - 1), pval_list)
        pvals.append(pval_list)
        # print pvals
        # print pval_list
        # raw_input()
    # results presentation
    # print(model.get_config())
    print 'P-Values:'
    print '\t'.join(['Accuracy', 'F1', 'Sensitivity', 'Specificity'])
    for s in pvals:
        print('%.4f\t%.4f\t%.4f\t%.4f' % (s[0], s[1], s[2], s[3]))
    print '\t'.join(
        ['Mean Accuracy', 'Mean F1', 'Mean Sensitivity', 'Mean Specificity'])
    print('%.4f\t%.4f\t%.4f\t%.4f' % (np.mean([s[0] for s in pvals]),
                                      np.mean([s[1] for s in pvals]),
                                      np.mean([s[2] for s in pvals]),
                                      np.mean([s[3] for s in pvals]), ))
    print len(score_list) - 1


def demo_seq2label_lstm_overdata(n_timesteps=6, n_stages=0, n_samples=0):
    """Sequence to label classification using LSTM
        Description: using oversampled data
        Input:
            n_timesteps: the length of sequences (# of time steps)
            n_stages: ahead of convertion
    """
    # model paramters
    nb_epochs = 200
    hidden_units = 128
    ndim = 630
    # model structure
    model = Sequential()
    model.add(
        LSTM(
            hidden_units,
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
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    res = []
    for n_sample in range(n_samples):
        if n_stages == 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_sample))
        elif n_sample >= 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_stages, n_sample))
        else:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
                (n_timesteps, n_stages))
        # raw_input()

        # model training and testing
        score_list = []
        for it in range(len(dataset)):
            X_train, y_train, X_test, y_test = \
                    dataset[it][0],dataset[it][1], dataset[it][2], dataset[it][3]
            X_train = np.asarray(X_train, dtype='float32')
            X_test = np.asarray(X_test, dtype='float32')
            y_train = np.asarray(y_train, dtype='int32')
            y_test = np.asarray(y_test, dtype='int32')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            print('Fold %d Training and Evaluation...' % (it + 1))
            model.fit(X_train,
                      y_train,
                      nb_epoch=nb_epochs,
                      show_accuracy=True,
                      verbose=1,
                      validation_split=0.1)
            y_predict = model.predict_classes(X_test)
            y_predict = [l[0] for l in y_predict]
            fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
            cid = list(thresholds).index(1)
            scores = [accuracy_score(y_test, y_predict),
                      f1_score(y_test, y_predict), tpr[cid], 1 - fpr[cid]]

            score_list.append(scores)
            # json_string = model.to_json()
            # open(data_info['path'] +
            #      'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' %
            #      (n_timesteps, it), 'w').write(json_string)
            # model.save_weights(
            #     data_info['path'] +
            #     'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' % (
            #         n_timesteps, it))

        # results presentation
        print(model.get_config())
        print 'LSTM Results:'
        print '\t'.join(['Accuracy', 'F1', 'Sensitivity', 'Specificity'])
        for s in score_list:
            print('%.4f\t%.4f\t%.4f\t%.4f' % (s[0], s[1], s[2], s[3]))
        print '\t'.join(['Mean Accuracy', 'Mean F1', 'Mean Sensitivity',
                         'Mean Specificity'])
        print('%.4f\t%.4f\t%.4f\t%.4f' %
              (np.mean([s[0] for s in score_list]),
               np.mean([s[1] for s in score_list]),
               np.mean([s[2] for s in score_list]),
               np.mean([s[3] for s in score_list]), ))
        res.append([np.mean([s[0] for s in score_list]),
                    np.mean([s[1] for s in score_list]),
                    np.mean([s[2] for s in score_list]),
                    np.mean([s[3] for s in score_list])])
    print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' %
          (np.mean([s[0] for s in res]), np.std([s[0] for s in res]),
           np.mean([s[1] for s in res]), np.std([s[1] for s in res]),
           np.mean([s[2] for s in res]), np.std([s[2] for s in res]),
           np.mean([s[3] for s in res]), np.std([s[3] for s in res])))


def demo_seq2label_cnn_overdata(n_timesteps=6, n_stages=0, n_samples=0):
    """Sequence to label classification using SVM
        Description: using oversampled data
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """

    # model paramters
    # if n_stages == 0:
    #     dataset = load_over_data(
    #         name_pkl='mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps)
    # else:
    #     dataset = load_over_data(
    #         name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
    #         (n_timesteps, n_stages))

    # batch_size = 16
    nb_epoch = 200
    # input image dimensions
    img_rows, img_cols = 10, n_timesteps * 63
    input_shape = (img_rows, img_cols, 1)  #len(dataset[0][0]),
    # number of convolutional filters to use
    nb_filters = 4
    filter_length = 2
    hidden_dims = 32
    # nb_classes=2
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # if K.image_dim_ordering() == 'th':
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(
        Convolution2D(
            nb_filters,
            kernel_size[0],
            kernel_size[1],
            border_mode='valid',
            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    #
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))

    model.summary()
    # model.compile(
    #     loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # model training and testing
    res = []
    for n_sample in range(n_samples):
        if n_stages == 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_sample))
        elif n_sample >= 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_stages, n_sample))
        else:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
                (n_timesteps, n_stages))

        score_list = []
        for it in range(len(dataset)):
            X_train, y_train, X_test, y_test = \
                    dataset[it][0],dataset[it][1], dataset[it][2], dataset[it][3]
            X_train = np.asarray(X_train, dtype='float32')
            X_test = np.asarray(X_test, dtype='float32')
            y_train = np.asarray(y_train, dtype='int32')
            y_test = np.asarray(y_test, dtype='int32')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            #
            # # img_rows, img_cols = X_train.shape[1], X_train.shape[2]
            # X_train = np.asarray(X_train, dtype='float32')
            # # X_train = X_train.reshape((X_train.shape[0], -1))
            # X_test = np.asarray(X_test, dtype='float32')
            # # X_test = X_test.reshape((X_test.shape[0], -1))
            #
            # # if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

            X_train /= 100
            X_test /= 100
            # input_shape = (1, img_rows, img_cols)
            # else:
            #     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            # input_shape = (img_rows, img_cols, 1)

            # y_train = np.asarray(y_train, dtype='int32')
            # y_test = np.asarray(y_test, dtype='int32')

            # Y_train = np_utils.to_categorical(y_train, nb_classes)
            # Y_test = np_utils.to_categorical(y_test, nb_classes)
            # print X_test
            # print Y_test
            # print np.amax(X_train)
            # print np.amax(X_test)

            # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
            # raw_input('wait')
            print('Fold %d Training and Evaluation...' % (it + 1))
            model.fit(X_train,
                      y_train,
                      #   batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      show_accuracy=True,
                      validation_split=0.1,
                      verbose=1)  #validation_data=(X_test, y_test)
            #   validation_data=(X_test, Y_test))
            # model.fit(X_train, y_train)
            # y_predict = model.predict(X_test)
            y_predict = model.predict_classes(X_test)
            y_predict = [l[0] for l in y_predict]
            fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
            cid = list(thresholds).index(1)
            scores = [accuracy_score(y_test, y_predict),
                      f1_score(y_test, y_predict), tpr[cid], 1 - fpr[cid]]
            score_list.append(scores)
            print scores
            # raw_input()

        # results presentation
        print 'CNN Results:'
        print '\t'.join(['Accuracy', 'F1', 'Sensitivity', 'Specificity'])
        for s in score_list:
            print('%.4f\t%.4f\t%.4f\t%.4f' % (s[0], s[1], s[2], s[3]))
        print '\t'.join(
            ['Mean Accuracy', 'Mean F1', 'Mean Sensitivity', 'Mean Specificity'])
        print('%.4f\t%.4f\t%.4f\t%.4f' % (np.mean([s[0] for s in score_list]),
                                          np.mean([s[1] for s in score_list]),
                                          np.mean([s[2] for s in score_list]),
                                          np.mean([s[3] for s in score_list]), ))
        res.append([np.mean([s[0] for s in score_list]),
                    np.mean([s[1] for s in score_list]),
                    np.mean([s[2] for s in score_list]),
                    np.mean([s[3] for s in score_list])])
    print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' %
          (np.mean([s[0] for s in res]), np.std([s[0] for s in res]),
           np.mean([s[1] for s in res]), np.std([s[1] for s in res]),
           np.mean([s[2] for s in res]), np.std([s[2] for s in res]),
           np.mean([s[3] for s in res]), np.std([s[3] for s in res])))


def demo_seq2label_svm_overdata(n_timesteps=6, n_stages=0, n_samples=0):
    """Sequence to label classification using CNN
        Description: using oversampled data
        Input:
            n_timesteps: the length of sequences (# of time steps)
    """

    # model paramters
    # if n_stages == 0:
    #     dataset = load_over_data(
    #         name_pkl='mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps)
    # else:
    #     dataset = load_over_data(
    #         name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
    #         (n_timesteps, n_stages))


    res = []
    for n_sample in range(n_samples):
        if n_stages == 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_sample))
        elif n_sample >= 0:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv_sample%d.p' %
                (n_timesteps, n_stages, n_sample))
        else:
            dataset = load_over_data(
                name_pkl='mci2ad_%dt_%ds_prediction_over2_shuf_cv.p' %
                (n_timesteps, n_stages))
        model = SVC()
        # model training and testing
        score_list = []
        for it in range(len(dataset)):
            X_train, y_train, X_test, y_test = \
                    dataset[it][0],dataset[it][1], dataset[it][2], dataset[it][3]
            X_train = np.asarray(X_train, dtype='float32')
            X_train = X_train.reshape((X_train.shape[0], -1))
            X_test = np.asarray(X_test, dtype='float32')
            X_test = X_test.reshape((X_test.shape[0], -1))
            y_train = np.asarray(y_train, dtype='int32')
            y_test = np.asarray(y_test, dtype='int32')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            # raw_input('wait')
            print('Fold %d Training and Evaluation...' % (it + 1))
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            # y_predict = model.predict_classes(X_test)
            # y_predict = [l[0] for l in y_predict]
            fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
            cid = list(thresholds).index(1)
            scores = [accuracy_score(y_test, y_predict),
                      f1_score(y_test, y_predict), tpr[cid], 1 - fpr[cid]]
            # fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
            # # print 1-fpr, tpr, thresholds
            # scores = [accuracy_score(y_test, y_predict),
            #           f1_score(y_test, y_predict), tpr[1], 1 - fpr[1]]
            score_list.append(scores)
            # print scores
            # raw_input()

        # results presentation
        print 'SVM Results:'
        print '\t'.join(['Accuracy', 'F1', 'Sensitivity', 'Specificity'])
        for s in score_list:
            print('%.4f\t%.4f\t%.4f\t%.4f' % (s[0], s[1], s[2], s[3]))
        print '\t'.join(
            ['Mean Accuracy', 'Mean F1', 'Mean Sensitivity', 'Mean Specificity'])
        print('%.4f\t%.4f\t%.4f\t%.4f' % (np.mean([s[0] for s in score_list]),
                                          np.mean([s[1] for s in score_list]),
                                          np.mean([s[2] for s in score_list]),
                                          np.mean([s[3] for s in score_list]), ))
        res.append([np.mean([s[0] for s in score_list]),
                    np.mean([s[1] for s in score_list]),
                    np.mean([s[2] for s in score_list]),
                    np.mean([s[3] for s in score_list])])
    print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' %
          (np.mean([s[0] for s in res]), np.std([s[0] for s in res]),
           np.mean([s[1] for s in res]), np.std([s[1] for s in res]),
           np.mean([s[2] for s in res]), np.std([s[2] for s in res]),
           np.mean([s[3] for s in res]), np.std([s[3] for s in res])))


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
        model = model_from_json(
            open(data_info['path'] +
                 'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' % (
                     n_timesteps, it)).read())
        model.load_weights(
            data_info['path'] +
            'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' % (
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
            model = model_from_json(
                open(data_info['path'] +
                     'mci2ad_%dt_prediction_over_cv%d_model_tf_sparse.json' % (
                         n_timesteps, it)).read())
            model.load_weights(
                data_info['path'] +
                'mci2ad_%dt_prediction_over_cv%d_model_weights_tf_sparse.h5' %
                (n_timesteps, it))
            for layer in model.layers:
                #             print layer.get_config()
                #             raw_input('wait')
                if layer.get_config()['name'] == 'lstm_1':
                    g = layer.get_config()
                    h = layer.get_weights()
                    #                 print h
                    #                 raw_input('wait')
                    h1 = np.mean(
                        h[0::3], axis=0)  # three matrices at each time step
                    score = _region_rank(h1)
                    region_score = map(add, score,
                                       region_score)  # add values in two lists
        except IOError:
            pass

    region_score = np.asarray(region_score)  # score for all regions
    sumofscore = np.sum(region_score)
    #     print region_score
    #     print sumofscore
    #     raw_input('wait')
    dfile = 'RegionMap90a.txt'
    dict_region = dict(
        csv.reader(
            open(data_info['path'] + dfile, 'rb'), delimiter=' '))
    region_scoredict = dict()
    for region_id in range(len(region_score)):
        if dict_region[str(region_id)] in region_scoredict:
            region_scoredict[dict_region[str(region_id)]] += float(
                region_score[region_id]) / float(sumofscore)
        else:
            region_scoredict[dict_region[str(region_id)]] = float(region_score[
                region_id]) / float(sumofscore)
    sorted_region_scoredict = sorted(
        region_scoredict.items(), key=operator.itemgetter(1), reverse=True)
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


def main():
    """Main entry of the demo
    """
    print('Welcome to our demo!')
    print(
        'Please input method type: 1:seq2label 2:seq2seq 3:seq2label+sae 4:seq2seq+sae 5:analysis 6:seq2score'
    )
    demoid = int(raw_input())
    if demoid == 1:
        n_timesteps, n_stages, n_samples = 1, 4, 10
        # demo_seq2label_lstm(n_timesteps=1)
        demo_seq2label_svm_overdata(
            n_timesteps=n_timesteps, n_stages=n_stages, n_samples=n_samples)
        # demo_seq2label_svm_overdata(n_timesteps=4,n_stages=1)
        # demo_seq2label_cnn_overdata(n_timesteps=n_timesteps, n_stages=n_stages)
        print 'n_timesteps:%d n_stages:%d n_samples:%d'%(n_timesteps, n_stages, n_samples)
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
    elif demoid == 7:
        n_timesteps, n_stages = 5, 0
        power_estimation(n_timesteps=n_timesteps, n_stages=n_stages)
        print n_timesteps, n_stages
    else:
        print('demo type not supported')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print '--- %s seconds ---' % (time.time() - start_time)
