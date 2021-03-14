import numpy as np
import scipy.signal as sps
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
    
    def max_pooling_forward(self,z,stride):
         (N,C,H,W) = z.shape
         h = int(H/stride)
         w = int(W/stride)
            
         pool_z = np.zeros([N,C,h,w])
         for n in range (N):
             for c in range (C):
                 for d in range (h):
                     for e in range (w):
                            pool_z[n,c,d,e] = np.max(z[n,c,d*stride:d*stride+2,e*stride:e*stride+2])
                          
                     return pool_z
    def conv_forward(self,z,K,b,padding):
        padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        size = np.array(np.shape(padding_z))
        size.flatten()
        (N,C,H,W) = z.shape
       # y = np.zeros([N,C,H,W])
        D, C, k1, k2 = K.shape
        y = np.zeros([N,D,H,W])
       
        for i in range(size[0]):
            for j in range(D):
                for k in range(size[2]-2):                 
                    for l in range(size[3]-2):
                        #y[i,j,k,l] += sum(sum(padding_z[i,j,(k-1):(k+2),(l-1):(l+2)]*K[j,0,:,:]))
                        y[i,j,k,l] =  np.sum(padding_z[i, :, k:k + k1, l:l + k2] * K[j, :]) + b[j]
       
            

        return y
    pass




