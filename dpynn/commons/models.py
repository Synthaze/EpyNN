#
#
import numpy as np
from epynn.commons.models import dataSet


class Bag:

    def __init__(self, name='Bag'):

        self.name = name
        self.models = []

        return None

    def add_model(self, model):

        self.models.append(model)

        return None

    def predict(self, X_data, X_encode=False, X_scale=False):
        """Perform bag prediction of label from unlabeled samples in dataset.

        :param X_data: Set of sample features.
        :type X_data: list[list[int or float or str]] or :class:`numpy.ndarray`

        :param X_encode: One-hot encode sample features, defaults to `False`.
        :type X_encode: bool, optional

        :param X_scale: Normalize sample features within [0, 1] along all axis, default to `False`.
        :type X_scale: bool, optional

        :return: Data embedding and output of forward propagation.
        :rtype: :class:`epynn.commons.models.dataSet`
        """
        self.dsets = []

        for i, model in enumerate(self.models):

            dset = model.predict(X_data, X_encode, X_scale)

            del model.layers

            self.dsets.append(dset)

        self.dset = {}

        self.dset['ids'] = self.dsets[0].ids
        self.dset['P'] = np.zeros_like(self.dsets[0].P)

        for dset in self.dsets:
            self.dset['P'] += dset.P

        self.dset['P'] = self.dset['P'] / len(self.dsets)

        self.dset['E'] = 1.95 * np.sqrt(self.dset['P'] * (1-self.dset['P']) / len(self.dsets))

        return None
