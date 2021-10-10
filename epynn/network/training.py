# EpyNN/epynn/network/train.py


def model_training(model):
    """Perform the training of the Neural Network.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    # Iterate over training epochs
    for model.e in range(model.e, model.epochs):

        # Shuffle dtrain and prepare new batches
        model.embedding.training_batches()

        # Iterate over training batches
        for batch in model.embedding.batch_dtrain:

            # Pass through every layer.forward() methods
            A = model.forward(batch.X)

            # Compute derivative of loss
            dA = model.training_loss(batch.Y, A, deriv=True)

            # Pass through every layer.backward() methods
            model.backward(dA)

            # Accuracy and cost for batch
            model.batch_report(batch, A)

        # Selected metrics and costs for dsets
        model.evaluate()

        # Tabular report for dsets
        model.report()

    return None
