# -*- coding: utf-8 -*-

# from __future__ import print_function
import pandas as pd
import os
from os import walk
import re
import numpy as np
import cPickle as pkl
import sys
from operator import itemgetter

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle

from config import data_info

np.random.seed(1337)  # for reproducibility


def data_preproc(ipath, opath):
    """Change raw data format such that each patient will have one csv
        Input:
            ipath: path of input raw data files
            opath: path of output data files
    """
    print('loading data...')
    for (dirpath, dirs, files) in walk(ipath):
        print dirpath
        print files
        print dirs
        print len(files)
    oid = ''
    dfs = []
    for fn in sorted(files):
        cid = id_extract(fn)
        if cid is not None:
            df = pd.read_csv(ipath + fn, sep='\t', header=None)
            df.drop(df.columns[[-1]], inplace=True, axis=1)
            if cid != oid:
                if len(dfs) == 4:
                    print 'save id: ' + oid
                    cdf = pd.concat(dfs, axis=1, ignore_index=True)
                    cdf.to_csv(opath + oid + '.csv', header=False, index=False)
                    dfs[:] = []
                    print cdf.shape
                    x = cdf.values.tolist()
                    print type(x), len(x), len(x[0])
                    raw_input('wait')
                else:
                    print len(dfs)
                # start new id
                oid = cid
            dfs.append(df)
    if len(dfs) == 4:
        print 'save id: ' + oid
        cdf = pd.concat(dfs, axis=1, ignore_index=True)
        cdf.to_csv(opath + oid + '.csv', header=False, index=False)
        dfs[:] = []

    print('data successfully loaded')


def label_preproc(ipath='/home/zhen/Projects/Data/MCI/',
                  ifile='ADNILongitudinalDataInfo_MCI_Con.txt',
                  opath='/home/zhen/Projects/Data/MCI/SeqLabelsConv/'):
    """Extract labels from raw file
        Input:
            ipath: path of input files
            ifile: path of label file
            opath: path of output files
    """

    with open(ipath + ifile, 'r') as f:
        lines = f.readlines()
        for iline in lines:
            patid = id_extract(iline,
                               id_pattern=re.compile(ur'^\[Subj. ID: (\d+)\]'))
            if patid is not None:
                cid = patid
            if iline.startswith('Current'):
                words = iline.split()
                for i, v in enumerate(words):
                    if v == 'MCI':
                        words[i] = 0
                    elif v == 'AD' or v == '-':
                        words[i] = 1
                # print words
                writer = open(opath + str(cid).zfill(4) + '.csv', 'w')
                writer.write(' '.join(str(x) for x in words[1:7]))
                writer.close()
            else:
                continue

def score_preproc(ipath='/home/zhen/Projects/Data/MCI/',
                  ifiles=['ADNIDataInfo_ClinicalScores_MCI_Convert.txt','ADNIDataInfo_ClinicalScores_MCI_Non_Convert.txt'],
                  opath='/home/zhen/Projects/Data/MCI/'):
    """Extract scores from raw file
        Input:
            ipath: path of input files
            ifile: path of label file
            opath: path of output files
    """

    mmse_score_dict = {}
    mmse_pos = 5
    id_pos = 0
    time_pos=1
    for ifile in ifiles:
        with open(ipath + ifile, 'r') as f:
            lines = f.readlines()
            for iline in lines:
                words = iline.split()
                if not int(words[id_pos]) in mmse_score_dict:
                    mmse_score_dict[int(words[id_pos])]={int(words[time_pos]):float(words[mmse_pos])}
                else:
                    mmse_score_dict[int(words[id_pos])].update({int(words[time_pos]):float(words[mmse_pos])})
    pkl.dump(mmse_score_dict, open(opath + 'mmse_score_dict.p', 'wb'))


def save_data_with_labels():
    """Save full MCI data in pkl format with sequential labels
    """

    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    n_timesteps = 6
    n_dim = 630
    shape = (n_timesteps, n_dim)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass
    X = []  # data ndarray
    y = []  # label vector
    for f in fs1:
        z = np.zeros(shape, dtype=float)
        x = np.genfromtxt(path_conv + f, delimiter=',')
        z[:x.shape[0], :x.shape[1]] = x
        X.append(z)
        y1 = np.genfromtxt(path_convlabels + f, delimiter=' ')
        y.append(y1)
    for f in fs0:
        z = np.zeros(shape, dtype=float)
        x = np.genfromtxt(path_ncon + f, delimiter=',')
        z[:x.shape[0], :x.shape[1]] = x
        X.append(z)
        # X.append(np.genfromtxt(path_conv+f,delimiter=','))
        y.append(np.zeros(n_timesteps))
    X = np.asarray(X)
    y = np.asarray(y)
    X, y = shuffle(X, y, random_state=0)
    print y
    kf = KFold(len(y), n_folds=5)
    # print kf
    # raw_input('wait')
    pkl.dump([X, y, kf], open(path_pkl + 'seq_mci_shuf_cv.p', 'wb'))


def save_prediction_data(n_timesteps=5):
    """Save full MCI data in pkl format
        * for prediction (predict whether a patient will convert no matter how
        * soon)
    """
    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    name_pkl = 'mci2ad_%dt_prediction_shuf_cv.p' % n_timesteps
    n_dim = 630
    shape = (n_timesteps, n_dim)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass
    X = []  # data ndarray
    y = []  # label vector
    for f in fs1:
        y1 = np.genfromtxt(path_convlabels + f, delimiter=' ')
        if y1[n_timesteps - 1] == 0:
            z = np.zeros(shape, dtype=float)
            x0 = np.genfromtxt(path_conv + f, delimiter=',')
            x = x0[:n_timesteps, :]
            z[:x.shape[0], :x.shape[1]] = x
            X.append(z)
            y.append(1)
    for f in fs0:
        z = np.zeros(shape, dtype=float)
        x0 = np.genfromtxt(path_ncon + f, delimiter=',')
        x = x0[:n_timesteps, :]
        z[:x.shape[0], :x.shape[1]] = x
        X.append(z)
        # X.append(np.genfromtxt(path_conv+f,delimiter=','))
        y.append(0)
    X = np.asarray(X)
    y = np.asarray(y)
    X, y = shuffle(X, y, random_state=0)
    print 'time steps: %d' % n_timesteps
    print X.shape, y.shape, np.sum(y), len(y) - np.sum(y)
    kf = KFold(len(y), n_folds=5)
    # print kf
    # raw_input('wait')
    pkl.dump([X, y, kf], open(path_pkl + name_pkl, 'wb'))

def save_regression_data(n_timesteps=5):
    """Save full MCI data in pkl format
        * for prediction of score (predict the mmse score of the patient soon)
    """
    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    score_pkl = 'mmse_score_dict.p'
    name_pkl = 'mci2ad_%dt_regression_shuf_cv.p' % n_timesteps
    n_dim = 630
    shape = (n_timesteps, n_dim)
    # load mmse scores
    with open(path_pkl + score_pkl, 'rb') as f:
        mmse_score_dict = pkl.load(f)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass
    X = []  # data ndarray
    y = []  # label vector
    for f in fs1:
        pid = id_extract(f)
        if int(pid) in mmse_score_dict:
            if n_timesteps in mmse_score_dict[int(pid)]:
                if mmse_score_dict[int(pid)][n_timesteps] >=0:
                    z = np.zeros(shape, dtype=float)
                    x0 = np.genfromtxt(path_conv + f, delimiter=',')
                    x = x0[:n_timesteps, :]
                    z[:x.shape[0], :x.shape[1]] = x
                    X.append(z)
                    y.append(mmse_score_dict[int(pid)][n_timesteps])
    print len(X), len(y)
    for f in fs0:
        pid = id_extract(f)
        if int(pid) in mmse_score_dict:
            if n_timesteps in mmse_score_dict[int(pid)]:
                if mmse_score_dict[int(pid)][n_timesteps] >=0:
                    z = np.zeros(shape, dtype=float)
                    x0 = np.genfromtxt(path_ncon + f, delimiter=',')
                    x = x0[:n_timesteps, :]
                    z[:x.shape[0], :x.shape[1]] = x
                    X.append(z)
                    y.append(mmse_score_dict[int(pid)][n_timesteps])
    print len(X), len(y)
    X = np.asarray(X,dtype='float32')
    y = np.asarray(y,dtype='float32')
    X, y = shuffle(X, y, random_state=0)
    print 'time steps: %d' % n_timesteps
    print X.shape, y.shape
    kf = KFold(len(y), n_folds=5)
    # print kf
    # raw_input('wait')
    pkl.dump([X, y, kf], open(path_pkl + name_pkl, 'wb'))

def save_regression_dif_data(n_timesteps=5):
    """Save full MCI data in pkl format
        * for prediction of score (predict the mmse score of the patient soon)
    """
    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    score_pkl = 'mmse_score_dict.p'
    name_pkl = 'mci2ad_%dt_regression_dif_shuf_cv.p' % n_timesteps
    n_dim = 630
    shape = (n_timesteps, n_dim)
    # load mmse scores
    with open(path_pkl + score_pkl, 'rb') as f:
        mmse_score_dict = pkl.load(f)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass
    X = []  # data ndarray
    y = []  # label vector
    for f in fs1:
        pid = id_extract(f)
        if int(pid) in mmse_score_dict:
            if n_timesteps in mmse_score_dict[int(pid)] and n_timesteps-1 in mmse_score_dict[int(pid)]:
                if mmse_score_dict[int(pid)][n_timesteps] >=0 and mmse_score_dict[int(pid)][n_timesteps-1] >=0:
                    z = np.zeros(shape, dtype=float)
                    x0 = np.genfromtxt(path_conv + f, delimiter=',')
                    x = x0[:n_timesteps, :]
                    z[:x.shape[0], :x.shape[1]] = x
                    X.append(z)
                    y.append(mmse_score_dict[int(pid)][n_timesteps-1] - mmse_score_dict[int(pid)][n_timesteps])
    print len(X), len(y)
    for f in fs0:
        pid = id_extract(f)
        if int(pid) in mmse_score_dict:
            if n_timesteps in mmse_score_dict[int(pid)] and n_timesteps-1 in mmse_score_dict[int(pid)]:
                if mmse_score_dict[int(pid)][n_timesteps] >=0 and mmse_score_dict[int(pid)][n_timesteps-1] >=0:
                    z = np.zeros(shape, dtype=float)
                    x0 = np.genfromtxt(path_ncon + f, delimiter=',')
                    x = x0[:n_timesteps, :]
                    z[:x.shape[0], :x.shape[1]] = x
                    X.append(z)
                    y.append(mmse_score_dict[int(pid)][n_timesteps-1] - mmse_score_dict[int(pid)][n_timesteps])
    print len(X), len(y)
    X = np.asarray(X,dtype='float32')
    y = np.asarray(y,dtype='float32')
    X, y = shuffle(X, y, random_state=0)
    print 'time steps: %d' % n_timesteps
    print X.shape, y.shape
    kf = KFold(len(y), n_folds=5)
    # print kf
    # raw_input('wait')
    pkl.dump([X, y, kf], open(path_pkl + name_pkl, 'wb'))

def save_prediction_data_oversample(n_timesteps=5):
    """Save full MCI data in pkl format
        * for prediction (predict whether a patient will convert no matter how
        * soon)
        * oversample the positive examples utill achieve a balanced dataset
        * step 1: obtain original data
        * step 2: obtain cross-validate data
        * step 3: oversample the dataset to make it balanced
    """
    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    name_pkl = 'mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps # over2 is different from over
    n_dim = 630
    shape = (n_timesteps, n_dim)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass

    # obtain the raw data
    Xp = []  # data ndarray for positive
    yp = []  # label vector for positive
    Xn = []  # data ndarray for negative
    yn = []  # label vector for negative
    for f in fs1:
        y1 = np.genfromtxt(path_convlabels + f, delimiter=' ')
        if y1[n_timesteps - 1] == 0:
            z = np.zeros(shape, dtype=float)
            x0 = np.genfromtxt(path_conv + f, delimiter=',')
            x = x0[:n_timesteps, :]
            z[:x.shape[0], :x.shape[1]] = x
            Xp.append(z)
            yp.append(1)
    for f in fs0:
        z = np.zeros(shape, dtype=float)
        x0 = np.genfromtxt(path_ncon + f, delimiter=',')
        x = x0[:n_timesteps, :]
        z[:x.shape[0], :x.shape[1]] = x
        Xn.append(z)
        # X.append(np.genfromtxt(path_conv+f,delimiter=','))
        yn.append(0)
    # Xp = np.asarray(Xp)
    # Xn = np.asarray(Xn)
    # yp = np.asarray(yp)
    # yn = np.asarray(yn)
    # while len(yn)>len(yp):
    #     yp += yp
    #     Xp += Xp

    # obtain the cv_index
    nb_folds = 5
    if len(Xp)<nb_folds:
        nb_folds = len(Xp)
    print 'folds: %d' % nb_folds
    kfp = KFold(len(Xp),n_folds=nb_folds) #
    kfn = KFold(len(Xn),n_folds=nb_folds)
    Xp_cvlist = []
    Xn_cvlist = []
    yp_cvlist = []
    yn_cvlist=[]
    Xpte_cvlist = []
    Xnte_cvlist = []
    ypte_cvlist = []
    ynte_cvlist=[]
    Xtr_cvlist=[]
    ytr_cvlist=[]
    Xte_cvlist=[]
    yte_cvlist=[]
    for train_idx, test_idx in kfp:
        # print type(train_idx),train_idx
        # print type(Xp)
        # print itemgetter(train_idx)(Xp)
        Xp_cvlist.append([Xp[i] for i in train_idx])
        yp_cvlist.append([yp[i] for i in train_idx])
        Xpte_cvlist.append([Xp[i] for i in test_idx])
        ypte_cvlist.append([yp[i] for i in test_idx])
    for train_idx, test_idx in kfn:
        Xn_cvlist.append([Xn[i] for i in train_idx])
        yn_cvlist.append([yn[i] for i in train_idx])
        Xnte_cvlist.append([Xn[i] for i in test_idx])
        ynte_cvlist.append([yn[i] for i in test_idx])
    for i in range(len(Xp_cvlist)):
        if len(Xp_cvlist[i])<0.8*len(Xn_cvlist[i]):
            Xp_cvlist[i] *= 2
            yp_cvlist[i] *=2
        _Xtr = Xp_cvlist[i]+Xn_cvlist[i]
        _ytr = yp_cvlist[i]+yn_cvlist[i]
        _Xte = Xpte_cvlist[i]+Xnte_cvlist[i]
        _yte = ypte_cvlist[i]+ynte_cvlist[i]
        _Xtr, _ytr = shuffle(_Xtr,_ytr, random_state=0)
        _Xte, _yte = shuffle(_Xte,_yte, random_state=0)
        Xtr_cvlist.append(_Xtr)
        ytr_cvlist.append(_ytr)
        Xte_cvlist.append(_Xte)
        yte_cvlist.append(_yte)
    dataset = zip(Xtr_cvlist, ytr_cvlist,Xte_cvlist,yte_cvlist)
    print type(dataset)
    # obtain oversampled data

    # X = np.asarray(Xp+Xn)
    # y = np.asarray(yp+yn)
    # X, y = shuffle(X, y, random_state=0)
    # print 'time steps: %d' % n_timesteps
    # print X.shape, y.shape, np.sum(y), len(y) - np.sum(y)
    # kf = KFold(len(y), n_folds=5)
    # # print kf
    # # raw_input('wait')
    pkl.dump(dataset, open(path_pkl + name_pkl, 'wb'))


def save_stage_prediction_data_oversample(n_timesteps=5, n_stage=1):
    """Save full MCI data in pkl format
        * for prediction
          * predict whether a patient will convert with:
            ## n_stage before convert
            ## n_timesteps of observations
        * oversample/repeat the positive examples utill achieve a balanced dataset
        * step 1: obtain original data
        * step 2: obtain cross-validate data
        * step 3: oversample the dataset to make it balanced
    """
    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_convlabels = '/home/zhen/Projects/Data/MCI/SeqLabelsConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    name_pkl = 'mci2ad_%dt_prediction_over2_shuf_cv.p' % n_timesteps # over2 is different from over
    n_dim = 630
    shape = (n_timesteps, n_dim)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass

    # obtain the raw data
    Xp = []  # data ndarray for positive
    yp = []  # label vector for positive
    Xn = []  # data ndarray for negative
    yn = []  # label vector for negative
    for f in fs1:
        y1 = np.genfromtxt(path_convlabels + f, delimiter=' ')
        if y1[n_timesteps - 1] == 0:
            z = np.zeros(shape, dtype=float)
            x0 = np.genfromtxt(path_conv + f, delimiter=',')
            x = x0[:n_timesteps, :]
            z[:x.shape[0], :x.shape[1]] = x
            Xp.append(z)
            yp.append(1)
    for f in fs0:
        z = np.zeros(shape, dtype=float)
        x0 = np.genfromtxt(path_ncon + f, delimiter=',')
        x = x0[:n_timesteps, :]
        z[:x.shape[0], :x.shape[1]] = x
        Xn.append(z)
        # X.append(np.genfromtxt(path_conv+f,delimiter=','))
        yn.append(0)
    # Xp = np.asarray(Xp)
    # Xn = np.asarray(Xn)
    # yp = np.asarray(yp)
    # yn = np.asarray(yn)
    # while len(yn)>len(yp):
    #     yp += yp
    #     Xp += Xp

    # obtain the cv_index
    nb_folds = 5
    if len(Xp)<nb_folds:
        nb_folds = len(Xp)
    print 'folds: %d' % nb_folds
    kfp = KFold(len(Xp),n_folds=nb_folds) #
    kfn = KFold(len(Xn),n_folds=nb_folds)
    Xp_cvlist = []
    Xn_cvlist = []
    yp_cvlist = []
    yn_cvlist=[]
    Xpte_cvlist = []
    Xnte_cvlist = []
    ypte_cvlist = []
    ynte_cvlist=[]
    Xtr_cvlist=[]
    ytr_cvlist=[]
    Xte_cvlist=[]
    yte_cvlist=[]
    for train_idx, test_idx in kfp:
        # print type(train_idx),train_idx
        # print type(Xp)
        # print itemgetter(train_idx)(Xp)
        Xp_cvlist.append([Xp[i] for i in train_idx])
        yp_cvlist.append([yp[i] for i in train_idx])
        Xpte_cvlist.append([Xp[i] for i in test_idx])
        ypte_cvlist.append([yp[i] for i in test_idx])
    for train_idx, test_idx in kfn:
        Xn_cvlist.append([Xn[i] for i in train_idx])
        yn_cvlist.append([yn[i] for i in train_idx])
        Xnte_cvlist.append([Xn[i] for i in test_idx])
        ynte_cvlist.append([yn[i] for i in test_idx])
    for i in range(len(Xp_cvlist)):
        if len(Xp_cvlist[i])<0.8*len(Xn_cvlist[i]):
            Xp_cvlist[i] *= 2
            yp_cvlist[i] *=2
        _Xtr = Xp_cvlist[i]+Xn_cvlist[i]
        _ytr = yp_cvlist[i]+yn_cvlist[i]
        _Xte = Xpte_cvlist[i]+Xnte_cvlist[i]
        _yte = ypte_cvlist[i]+ynte_cvlist[i]
        _Xtr, _ytr = shuffle(_Xtr,_ytr, random_state=0)
        _Xte, _yte = shuffle(_Xte,_yte, random_state=0)
        Xtr_cvlist.append(_Xtr)
        ytr_cvlist.append(_ytr)
        Xte_cvlist.append(_Xte)
        yte_cvlist.append(_yte)
    dataset = zip(Xtr_cvlist, ytr_cvlist,Xte_cvlist,yte_cvlist)
    print type(dataset)
    # obtain oversampled data

    # X = np.asarray(Xp+Xn)
    # y = np.asarray(yp+yn)
    # X, y = shuffle(X, y, random_state=0)
    # print 'time steps: %d' % n_timesteps
    # print X.shape, y.shape, np.sum(y), len(y) - np.sum(y)
    # kf = KFold(len(y), n_folds=5)
    # # print kf
    # # raw_input('wait')
    pkl.dump(dataset, open(path_pkl + name_pkl, 'wb'))



def save_data():
    """Save full MCI data in pkl format
    """

    path_conv = '/home/zhen/Projects/Data/MCI/DFConv/'
    path_ncon = '/home/zhen/Projects/Data/MCI/DFNConv/'
    path_pkl = '/home/zhen/Projects/Data/MCI/'
    n_timesteps = 6
    n_dim = 630
    shape = (n_timesteps, n_dim)
    for (r1, ds1, fs1) in walk(path_conv):
        pass
    for (r0, ds0, fs0) in walk(path_ncon):
        pass
    X = []  # data ndarray
    y = []  # label vector
    for f in fs1:
        z = np.zeros(shape, dtype=float)
        x = np.genfromtxt(path_conv + f, delimiter=',')
        z[:x.shape[0], :x.shape[1]] = x
        X.append(z)
        y.append(1)
    for f in fs0:
        z = np.zeros(shape, dtype=float)
        x = np.genfromtxt(path_ncon + f, delimiter=',')
        z[:x.shape[0], :x.shape[1]] = x
        X.append(z)
        # X.append(np.genfromtxt(path_conv+f,delimiter=','))
        y.append(0)
    X = np.asarray(X)
    y = np.asarray(y)
    X, y = shuffle(X, y, random_state=0)
    print y
    kf = KFold(len(y), n_folds=5)
    # print kf
    # raw_input('wait')
    pkl.dump([X, y, kf], open(path_pkl + 'mci_shuf_cv.p', 'wb'))


def load_full_data(path_pkl=data_info['path'],
                   name_pkl='mci_shuf_cv.p'):
    """Load full data from pickle file
    """

    with open(path_pkl + name_pkl, 'rb') as f:
        data = pkl.load(f)
    return data[0], data[1], data[2]

def load_over_data(path_pkl=data_info['path'],
                   name_pkl='mci2ad_5t_prediction_over2_shuf_cv.p'):
    """Load oversampled data from pickle file
    """

    with open(path_pkl + name_pkl, 'rb') as f:
        data = pkl.load(f)
    return data

def id_extract(s, id_pattern=re.compile(ur'^(\d+)')):
    """Extract number in the head of a string
    """
    m = re.findall(id_pattern, s)
    if len(m) > 0:
        return m[0]
    else:
        return None


def main():
    """function for runing some functionalities"""
    # save_prediction_data(n_timesteps=1)
    save_regression_data(n_timesteps=6)
    # save_regression_dif_data(n_timesteps=1)
    # save_prediction_data_oversample(n_timesteps=1)
    # score_preproc()


if __name__ == '__main__':
    """the scripts when run this file"""
    main()
    # data_preproc('/home/zhen/Projects/Data/MCI/Conv/','/home/zhen/Projects/Data/MCI/DFConv/')
    # data_preproc('/home/zhen/Projects/Data/MCI/Non_Conv/','/home/zhen/Projects/Data/MCI/DFNConv/')
    # id_extract('0030CCF.txt')
