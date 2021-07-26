#EpyNN/nnlibs/meta/train.py
import nnlibs.commons.library as cli
import nnlibs.commons.metrics as cme
import nnlibs.commons.logs as clo
import nnlibs.commons.io as cio

import nnlibs.meta.parameters as mp
import nnlibs.meta.backward as mb
import nnlibs.meta.forward as mf


def run_train(model,batch_dtrain):
    """An example docstring for a function definition."""

    model.initialize()

    epochs = model.se_hPars['training_epochs']

    for e in range(epochs):

        for batch in batch_dtrain:

            A = X = batch.X

            A = model.forward(A)

            dA = dX = A - batch.Y.T

            dA = model.backward(dA)

        model.compute_metrics()

        model.evaluate()

        model.logs()

    return None
