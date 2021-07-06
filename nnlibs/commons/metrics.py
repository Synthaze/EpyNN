#EpyNN/nnlibs/commons/metrics.py
import numpy as np


def compute_metrics(model,dsets,hPars,runData):

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

    for k, dset in enumerate(reversed(dsets)):

        A = dset.X.T

        dset.A = model.predict(A)

        dset.P = np.argmax(dset.A.T,axis=1)

        for s in runData.s.keys():

            m = metrics[s](dset,hPars)

            runData.s[s][len(dsets)-1-k].append(m)

    return runData


def accuracy(dset,hPars):
    accuracy = np.mean(dset.P - dset.y == 0)
    return accuracy


def recall(dset,hPars):

    oy = dset.P + dset.y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(dset.P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(dset.P) - tn

    recall = tp/(tp+fn+hPars.c['E'])

    return recall


def precision(dset,hPars):

    oy = dset.P + dset.y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(dset.P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(dset.P) - tn

    precision = tp/(tp+fp+hPars.c['E'])

    return precision


def CCE(dset,hPars):
    CCE = - 1 * np.mean(dset.Y * np.log(dset.A.T + hPars.c['E']))
    return CCE


def CE(dset,hPars):
    B = np.array(list(1.0 * (dset.A.T[i] == np.max(dset.A.T[i])) for i in range(dset.A.T.shape[0])))
    CE = np.sum(np.abs(B - dset.Y)) / len(dset.Y) / 2.0
    return CE


def BCE(dset,hPars):
    BCE = - np.mean(np.multiply(dset.Y, np.log(dset.A.T+hPars.c['E'])) + np.multiply((1-dset.Y), np.log(1-dset.A.T+hPars.c['E'])))
    return BCE


def MSE(dset,hPars):
    MSE = np.mean(np.square(np.subtract(dset.Y,dset.A.T)+hPars.c['E']))
    return MSE


def MAE(dset,hPars):
    MAE = np.mean(np.abs(dset.Y-dset.A.T))
    return MAE


def RMSLE(dset,hPars):
    RMSLE = np.sqrt(np.mean(np.square(np.log1p(dset.Y+hPars.c['E']) - np.log1p(dset.A.T+hPars.c['E']))))
    return RMSLE


def KLD(dset,hPars):
    KLD = np.mean(dset.A.T * np.log(((dset.A.T+hPars.c['E']) / (dset.Y+hPars.c['E']))))
    return KLD
