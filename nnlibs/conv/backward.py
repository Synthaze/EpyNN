#EpyNN/nnlibs/conv/backward.py
import numpy as np


def convolution_backward(layer,dZ):

    im, ih, iw, id = layer.s['X']

    dA_prev = np.zeros(layer.s['X'])

    dW = np.zeros(layer.s['W'])
    db = np.zeros(layer.s['b'])

    for t in range(layer.n_rows):

        row = dZ[:, t::layer.n_rows, :, :]

        for l in range(layer.n_cols):

            b = (ih - t * layer.d['s']) % layer.d['fw']
            r = (iw - l * layer.d['s']) % layer.d['fw']  # region of X and dZ for this block

            block = row[:, :, l * layer.d['s']::layer.n_cols, :]           # block = corresponding region of dA

            block = np.expand_dims(block, 3)              # axis for channels
            block = np.expand_dims(block, 3)              # axis for rows
            block = np.expand_dims(block, 3)              # axis for columns

            dW_block = block * layer.X_blocks[t][l]

            dW_block = np.sum(dW_block, 2)
            dW_block = np.sum(dW_block, 1)
            dW_block = np.sum(dW_block, 0)

            dW += dW_block

            db_block = np.sum(dW_block, 2, keepdims=True)
            db_block = np.sum(db_block, 1, keepdims=True)
            db_block = np.sum(db_block, 0, keepdims=True)

            db += db_block

            dA_prev_block = block * layer.p['W']
            dA_prev_block = np.sum(dA_prev_block, 6)
            dA_prev_block = np.reshape(dA_prev_block, (im, ih - b - t, iw - r - l, id))

            dA_prev[:, t:ih - b, l:iw - r, :] += dA_prev_block

    layer.g = { 'dW': dW, 'db': db}

    if layer.d['p'] > 0:
        p = layer.d['p']                             # remove padding
        dA_prev = dA_prev[:, p:-p, p:-p, :]

    return dA_prev
