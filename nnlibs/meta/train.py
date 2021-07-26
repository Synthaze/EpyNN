# EpyNN/nnlibs/meta/train.py


def model_training(model):

    model.initialize()

    for model.e in range(model.epochs):

        for batch in model.embedding.batch_dtrain:

            A = X = batch.X

            A = model.forward(A)

            dA = dX = A - batch.Y.T

            dA = model.backward(dA)

        model.compute_metrics()

        model.evaluate()

        model.logs()

    return None
