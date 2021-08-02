# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Zeros-output of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA

    dA = np.zeros(layer.fs['X'])

    return dX, dA


def pooling_backward(layer, dA):
    """Backward propagate signal to previous layer.
    """
    # (1) Initialize cache
    dX, dA = initialize_backward(layer, dA)

    # Loop through image rows
    for t in range(layer.d['R']):

        mask_row = layer.fc['Z'][:, t::layer.d['R'], :, :]

        row = dX[:, t::layer.d['R'], :, :]

        # Loop through row columns
        for l in range(layer.d['C']):

            # region of X and dZ for this block
            b = (layer.d['ih'] - t * layer.d['s']) % layer.d['w']
            # region of X and dZ for this block
            r = (layer.d['iw'] - l * layer.d['s']) % layer.d['w']
            # _
            mask = mask_row[:, :, l * layer.d['s']::layer.d['C'], :]
            mask = layer.assemble_block(mask, t, b, l, r)
            # _
            block = row[:, :, l * layer.d['s']::layer.d['C'], :]
            block = layer.assemble_block(block, t, b, l, r)
            # _
            mask = (layer.fc['X'][:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] == mask)
            # _
            dA[:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] += block * mask

    dA = layer.bc['dA'] = dA

    return dA    # To previous layer
