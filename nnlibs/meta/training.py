# EpyNN/nnlibs/meta/train.py


def model_training(model):
    """Perform the training of the Neural Network.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """
    for model.e in range(model.epochs):

        for batch in model.embedding.batch_dtrain:

            X = batch.X

            A = model.forward(X)
            
            dA = model.training_loss(batch.Y, A, deriv=True) / A.shape[1]

            dA = model.backward(dA)

        model.compute_metrics()

        model.evaluate()

        model.logs()

    return None
