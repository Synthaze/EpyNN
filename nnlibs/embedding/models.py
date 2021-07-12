#EpyNN/nnlibs/embedding/parameters.py
import nnlibs.embedding.parameters as ep
import nnlibs.embedding.backward as eb
import nnlibs.embedding.forward as ef

import nnlibs.settings as se


class Embedding:

    def __init__(self,dataset,se_dataset=se.dataset,encode=False):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True

        ### Define layer dictionaries attributes
        ## Dimensions
        self.d = {}
        ## Parameters
        self.p = {}
        ## Gradients
        self.g = {}
        ## Forward pass cache
        self.fc = {}
        ## Backward pass cache
        self.bc = {}
        ## Forward pass shapes
        self.fs = {}
        ## Backward pass shapes
        self.bs = {}

        ### Set keys for layer cache attributes
        self.attrs = ['X','A']

        ### Init shapes
        #tp.init_shapes(self)
        prefix = se_dataset['dataset_name']

        if encode == True:
            dataset = encoded_dataset = ep.encode_dataset(self,dataset)
        else:
            self.d['v'] = None

        dtrain, dtest, dval = ep.split_dataset(dataset,se_dataset)

        self.dtrain = ep.object_vectorize(dtrain,type='dtrain',prefix=prefix)
        self.dtest = ep.object_vectorize(dtest,type='dtest',prefix=prefix)
        self.dval = ep.object_vectorize(dval,type='dval',prefix=prefix)

        batch_dtrain = ep.mini_batches(dtrain,se_dataset)

        self.batch_dtrain = []

        for b, batch in enumerate(batch_dtrain):
            batch = ep.object_vectorize(batch,type='dtrain_'+str(b),prefix=prefix)
            self.batch_dtrain.append(batch)

    def init_shapes(self):
        ep.init_shapes(self)

    def forward(self,A):
        # Forward pass
        A = ef.embedding_forward(self,A)
        return A


    def backward(self,dA):
        # Backward pass
        dA = eb.embedding_backward(self,dA)
        return dA
