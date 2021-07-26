#EpyNN/nnlibs/commons/metrics.py
import numpy as np


def model_compute_metrics(model):

    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'CCE': CCE,
        'CE': CE,
        'BCE': BCE,
        'MSE': MSE,
        'MAE': MAE,
        'RMSLE': RMSLE,
        'KLD': KLD,
    }

    embedding = model.layers[0]

    dsets = [embedding.dtrain,embedding.dtest,embedding.dval]

    hPars = model.se_hPars

    for k, dset in enumerate(reversed(dsets)):

        A = X = dset.X

        dset.A = model.forward(X).T

        dset.P = np.argmax(dset.A,axis=1)

        for s in model.metrics.keys():

            m = metrics[s](dset,hPars)

            model.metrics[s][len(dsets)-1-k].append(m)

    return None


def accuracy(dset,hPars):
    accuracy = np.mean(dset.P - dset.y == 0)
    return accuracy


def recall(dset,hPars):

    oy = dset.P + dset.y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(dset.P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(dset.P) - tn

    recall = tp/(tp+fn+hPars['min_epsilon'])

    return recall


def precision(dset,hPars):

    oy = dset.P + dset.y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(dset.P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(dset.P) - tn

    precision = tp/(tp+fp+hPars['min_epsilon'])

    return precision


def CCE(dset,hPars):
    CCE = - 1 * np.mean(dset.Y * np.log(dset.A + hPars['min_epsilon']))
    return CCE


def CE(dset,hPars):
    B = np.array(list(1.0 * (dset.A[i] == np.max(dset.A[i])) for i in range(dset.A.shape[0])))
    CE = np.sum(np.abs(B - dset.Y)) / len(dset.Y) / 2.0
    return CE


def BCE(dset,hPars):
    BCE = - np.mean(np.multiply(dset.Y, np.log(dset.A+hPars['min_epsilon'])) + np.multiply((1-dset.Y), np.log(1-dset.A+hPars['min_epsilon'])))
    return BCE


def MSE(dset,hPars):
    MSE = np.mean(np.square(np.subtract(dset.Y,dset.A)+hPars['min_epsilon']))
    return MSE


def MAE(dset,hPars):
    MAE = np.mean(np.abs(dset.Y-dset.A))
    return MAE


def RMSLE(dset,hPars):
    RMSLE = np.sqrt(np.mean(np.square(np.log1p(dset.Y+hPars['min_epsilon']) - np.log1p(dset.A+hPars['min_epsilon']))))
    return RMSLE


def KLD(dset,hPars):
    KLD = np.mean(dset.A * np.log(((dset.A+hPars['min_epsilon']) / (dset.Y+hPars['min_epsilon']))))
    return KLD
