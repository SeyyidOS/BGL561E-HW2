from DL.bases import LayerWithWeights
import numpy as np
from copy import copy


class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')

        # Calculate output's H and W according to your lecture notes
        out_H = int(((H + 2*self.padding - FH) / self.stride) + 1)
        out_W = int(((W + 2*self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])

        # TO DO: Do cross-correlation by using for loops
        # YOUR CODE STARTS

        out = np.zeros([N, F, out_H, out_W])

        for n in range(N):  # batch
            for f in range(F):  # filters
                for i in range(out_H):  # height
                    for j in range(out_W):  # width
                        h_start = i * self.stride
                        h_end = h_start + FH
                        w_start = j * self.stride
                        w_end = w_start + FW
                        
                        out[n, f, i, j] = np.sum(
                            padded_x[n, :, h_start:h_end, w_start:w_end] * self.W[f, :, :, :]
                        ) + self.b[f]

        # YOUR CODE ENDS
        
        return out

    def backward(self, dprev):
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)


        # YOUR CODE STARTS
        # Bias
        for f in range(F):
            db[f] = np.sum(dprev[:, f, :, :])

        # Weights and Input
        for n in range(N):
            for f in range(F):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        h_end = h_start + FH
                        w_start = j * self.stride
                        w_end = w_start + FW

                        # Weights
                        dw[f, :, :, :] += padded_x[n, :, h_start:h_end, w_start:w_end] * dprev[n, f, i, j]

                        # Input
                        dx_temp[n, :, h_start:h_end, w_start:w_end] += self.W[f, :, :, :] * dprev[n, f, i, j]

        # Remove padding
        dx = dx_temp[:, :, self.padding:-self.padding, self.padding:-self.padding]


        # YOUR CODE ENDS

        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

