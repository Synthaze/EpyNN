# EpyNN/nnlibs/conv/backward.py
# Related third party imports
import numpy as np


def convolution_backward(layer, dA):

    dX, im, ih, iw, id, dA = initialize_backward(layer, dA)

    # Loop through image rows
    for t in range(layer.d['R']):

        row = dX[:, t::layer.d['R'], :, :]

        # Loop through row columns
        for l in range(layer.d['C']):

            # region of X and dZ for this block
            b = (ih - t * layer.d['s']) % layer.d['w']
            # region of X and dZ for this block
            r = (iw - l * layer.d['s']) % layer.d['w']

            # block = corresponding region of dA
            block = row[:, :, l * layer.d['s']::layer.d['C'], :]
            # Axis for channels, rows, columns
            block = np.expand_dims(block, 3)
            block = np.expand_dims(block, 3)
            block = np.expand_dims(block, 3)

            layer.bc['dXb'][t][l] = block

            dA_block = block * layer.p['W']
            dA_block = np.sum(dA_block, 6)
            dA_block = np.reshape(dA_block, (im, ih - b - t, iw - r - l, id))

            dA[:, t:ih - b, l:iw - r, :] += dA_block

    #dA = layer.bc['dA'] = restore_padding(layer,dA)

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    layer.bc['dXb'] = [ [[] for t in range(layer.d['R'])] for l in range(layer.d['C'])]

    im, ih, iw, id = layer.fs['X']

    dA = np.zeros(layer.fs['X'])

    return dX, im, ih, iw, id, dA

#
#
#
# def restore_padding(layer,dA):
#
#     if layer.d['p'] > 0:
#         p = layer.d['p']
#         dA = dA[:, p:-p, p:-p, :]
#
#     return dA
