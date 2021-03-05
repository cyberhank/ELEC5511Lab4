import numpy as np
class NeuralNetwork():
    def __init__(self):
        return
    def relu_forward(self,z):
        return np.maximum(0, z)
    def flatten_forward(self,z):
        N = z.shape[0]
        return np.reshape(z, (N, -1))
    def fc_forward(self, z, W, b):
        return np.dot(W, z.transpose()) + np.expand_dims(b,1).repeat(z.shape[0],axis = 1)
    def conv_forward(self, z, K, b):
        return conv_z
    def max_pooling_forward(self,z):
        return pool_z
    def padding(self,x, padding):
        y = []
        for i in range(np.size(x, axis=1)+2):
           for j in range(np.size(x, axis=0)+2):
               if i == 0:
                   y[i,j] = padding
               if i == np.size(x, axis=1)+2:
                    y[i,j] = padding
               if j == 0:
                   y[i,j] = padding
               if i == np.size(x, axis=0)+2:
                    y[i,j] = padding

        return y
    pass




