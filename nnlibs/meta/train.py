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

    hPars = model.hPars
    runData = model.runData

    for hPars.e in range(hPars.i):

        for batch in batch_dtrain:

            A = X = batch.X

            A = mf.forward(model,A)

            dA = A - batch.Y.T

            mb.backward(model,dA)

            mp.update_params(model,hPars)

        cme.compute_metrics(model,hPars,runData)

        cli.check_and_write(model,hPars,runData)

        clo.model_core_logs(model,hPars,runData)

    return None
