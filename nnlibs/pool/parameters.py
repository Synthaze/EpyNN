#EpyNN/nnlibs/pool/parameters.py


def init_shapes(layer,f_width,stride):
    ### Set layer dictionaries values
    ## Dimensions

    layer.d['fw'] = f_width
    layer.d['s'] = stride

    return None
