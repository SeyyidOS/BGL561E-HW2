import numpy as np
from DL.bases import Layer, LayerWithWeights


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        """
        :param x: activations/inputs from previous layer
        :return: output of affine layer
        """
        # Save x for using in backward pass
        self.x = x.copy()
        out = None

        ##### YOUR CODE STARTS #####

        # Vectorize the input to [batchsize, others] array
        x = x.reshape(x.shape[0], -1)  # [batchsize, -1]

        # Do the affine transform
        out = np.dot(x, self.W) + self.b

        # print(out)
        ##### YOUR CODE ENDS #######
        return out

    def backward(self, dprev):
        """
        :param dprev: gradient of next layer:
        :return: downstream gradient
        """

        batch_size = self.x.shape[0]
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        # Vectorize the input to a 1D ndarray
        x_vectorized = self.x.reshape(batch_size, -1)

        dx = np.dot(dprev, self.W.T).reshape(self.x.shape)  # dl/dx = dl/dout * dout/dx

        dw = np.dot(x_vectorized.T, dprev)  # dl/dw = dout/dw * dl/dout

        db = np.dot(dprev.T, np.ones(batch_size))  # dl/db = dl/dout * dout/db

        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return "Affine layer"
