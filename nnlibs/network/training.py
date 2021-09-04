# EpyNN/nnlibs/network/train.py


def model_training(model):
    """Perform the training of the Neural Network.

    :param model: An instance of EpyNN network object.
    :type model: :class:`nnlibs.network.models.EpyNN`
    """
    for model.e in range(model.e, model.epochs):

        model.embedding.training_batches()

        for batch in model.embedding.batch_dtrain:

            A = model.forward(batch.X)

            dA = model.training_loss(batch.Y, A, deriv=True)

            model.backward(dA)

            model.batch_report(batch, A)

        model.evaluate()

        model.report()

    return None
