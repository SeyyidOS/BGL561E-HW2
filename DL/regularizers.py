from DL.bases import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, p=.5):
        '''
            :param p: dropout factor
        '''
        self.mask = None
        self.mode = 'train'
        self.p = p

    def forward(self, x, seed=None):
        '''
            :param x: input to dropout layer
            :param seed: seed (used for testing purposes)
        '''
        if seed is not None:
            np.random.seed(seed)
            
        # YOUR CODE STARTS

        if self.mode == 'train':
            self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            out = x * self.mask / (1 - self.p)
        elif self.mode == 'test':
            out = x

        # YOUR CODE ENDS
        
        else:
            raise ValueError('Invalid argument!')
        
        return out

    def backward(self, dprev):
        dx = None
        
        # YOUR CODE STARTS

        dx = dprev * self.mask / (1 - self.p)

        # YOUR CODE ENDS
        
        return dx


class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = int(((H - self.pool_height) / self.stride) + 1)
        out_W = int(((W - self.pool_width) / self.stride) + 1)

        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])

        # YOUR CODE STARTS
        
        for n in range(N):  # batch
            for c in range(C):  # channels
                for i in range(out_H):  # height
                    for j in range(out_W):  # width
                        h_start = i * self.stride
                        h_end = h_start + self.pool_height
                        w_start = j * self.stride
                        w_end = w_start + self.pool_width

                        out[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])

        # YOUR CODE ENDS
        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)

        # Calculate the gradient (dx)
        # YOUR CODE STARTS
        
        for n in range(N):  # batch
            for c in range(C):  # channels
                for i in range(dprev_H):  # height
                    for j in range(dprev_W):  # width
                        h_start = i * self.stride
                        h_end = h_start + self.pool_height
                        w_start = j * self.stride
                        w_end = w_start + self.pool_width

                        x_window = x[n, c, h_start:h_end, w_start:w_end]
                        max_mask = (x_window == np.max(x_window))
                        dx[n, c, h_start:h_end, w_start:w_end] += dprev[n, c, i, j] * max_mask

        # YOUR CODE ENDS
        return dx


class AveragePool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = int(((H - self.pool_height) / self.stride) + 1)
        out_W = int(((W - self.pool_width) / self.stride) + 1)

        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])
        # YOUR CODE STARTS

        for n in range(N):  # batch
            for c in range(C):  # channels
                for i in range(out_H):  # height
                    for j in range(out_W):  # width
                        h_start = i * self.stride
                        h_end = h_start + self.pool_height
                        w_start = j * self.stride
                        w_end = w_start + self.pool_width

                        out[n, c, i, j] = np.mean(x[n, c, h_start:h_end, w_start:w_end])

        # YOUR CODE ENDS
        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)
        # YOUR CODE STARTS

        for n in range(N):  # batch
            for c in range(C):  # channels
                for i in range(dprev_H):  # height
                    for j in range(dprev_W):  # width
                        h_start = i * self.stride
                        h_end = h_start + self.pool_height
                        w_start = j * self.stride
                        w_end = w_start + self.pool_width

                        dx[n, c, h_start:h_end, w_start:w_end] += dprev[n, c, i, j] / (self.pool_height * self.pool_width)

        # YOUR CODE ENDS
        return dx


class BatchNorm(Layer):
    def __init__(self, D, momentum=.9):
        self.mode = 'train'
        self.normalized = None

        self.x_sub_mean = None
        self.momentum = momentum
        self.D = D
        self.running_mean = np.zeros(D)
        self.running_var = np.zeros(D)
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.ivar = np.zeros(D)
        self.sqrtvar = np.zeros(D)

    def forward(self, x, gamma=None, beta=None):
        if self.mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            if gamma is not None:
                self.gamma = gamma.copy()
            if beta is not None:
                self.beta = beta.copy()

            # Normalise our batch
            self.normalized = ((x - sample_mean) /
                               np.sqrt(sample_var + 1e-5)).copy()
            self.x_sub_mean = x - sample_mean

            # YOUR CODE STARTS
            
            # Update our running mean and variance then store.

            running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

            out = self.gamma * self.normalized + self.beta


            # YOUR CODE ENDS
            self.running_mean = running_mean.copy()
            self.running_var = running_var.copy()

            self.ivar = 1./np.sqrt(sample_var + 1e-5)
            self.sqrtvar = np.sqrt(sample_var + 1e-5)

            return out
        elif self.mode == 'test':
            out = None
            out = self.gamma * self.normalized + self.beta
            return out
        else:
            raise Exception(
                "INVALID MODE! Mode should be either test or train")
        
        
    def backward(self, dprev):
        N, D = dprev.shape
        # YOUR CODE STARTS

        dgamma = np.sum(dprev * self.normalized, axis=0)
        dbeta = np.sum(dprev, axis=0)

        dnormalized = dprev * self.gamma
        dvar = np.sum(dnormalized * self.x_sub_mean * -0.5 * (self.ivar**3), axis=0)
        dmean = np.sum(dnormalized * -self.ivar, axis=0) + dvar * np.mean(-2.0 * self.x_sub_mean, axis=0)

        dx = (dnormalized * self.ivar) + (dvar * 2.0 * self.x_sub_mean / N) + (dmean / N)


        # YOUR CODE ENDS
        return dx, dgamma, dbeta




class BatchNorm2d(Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.mode = 'train'
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize weights and biases
        self.gamma = np.ones((1,num_features,1,1))
        self.beta = np.zeros((1,num_features,1,1))

        # Initialize running mean and variance
        self.running_mean = np.zeros((1,num_features,1,1))
        self.running_var = np.ones((1,num_features,1,1))

        # Buffers to hold intermediate values during training
        self.x_sub_mean = None
        self.batch_mean = None
        self.batch_var = None
        self.x_normalized = None

    def forward(self, x, gamma=None, beta=None):
        # Ensure input shape is (batch_size, num_features, height, width)
        # Calculate mean and variance along batch and spatial dimensions
        if self.mode == 'train':
            # Calculate mean and variance over batch and spatial dimensions
            self.batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            if gamma is not None:
                self.gamma = gamma.copy()
            if beta is not None:
                self.beta = beta.copy()

            # Normalize input
            self.x_normalized = (x - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            self.x_sub_mean = x - self.batch_mean

            # YOUR CODE STARTS
            
            # Update our running mean and variance then store.

            running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            # YOUR CODE ENDS
            self.running_mean = running_mean.copy()
            self.running_var = running_var.copy()

        else:
            # During inference, use running mean and variance
            self.x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        out = self.gamma * self.x_normalized + self.beta
        return out

    def backward(self, dprev):
        N, C, H, W = dprev.shape
        dx, dgamma, dbeta = None, None, None

        # YOUR CODE STARTS

        # Gradients for gamma and beta
        dgamma = np.sum(dprev * self.x_normalized, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dprev, axis=(0, 2, 3), keepdims=True)

        # Gradient for normalized input
        dnormalized = dprev * self.gamma

        # Gradients for variance and mean
        dvar = np.sum(dnormalized * self.x_sub_mean * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=(0, 2, 3), keepdims=True) + \
                dvar * np.sum(-2 * self.x_sub_mean, axis=(0, 2, 3), keepdims=True) / (N * H * W)

        # Gradient for input
        dx = (dnormalized / np.sqrt(self.batch_var + self.epsilon)) + \
            (dvar * 2 * self.x_sub_mean / (N * H * W)) + \
            (dmean / (N * H * W))

        # YOUR CODE ENDS

        return dx, dgamma, dbeta
