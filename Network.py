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
    
    def max_pooling_forward(self,z,stride =2 ):
        (N,C,H,W) = z.shape # creating variables for the size of the input matrix
        h = int(H/stride) # scaling of the height
        w = int(W/stride) # scalling the width
            
        pool_z = np.zeros([N,C,h,w])
        for n in range (N): # iterates throught the input files
            for c in range (C): # iterates through the channels
                # the last two for-loops iterate through the convolved and relu corrected functions
                for d in range (h): 
                    for e in range (w):
                        pool_z[n,c,d,e] = np.max(z[n,c,d*stride:d*stride+2,e*stride:e*stride+2]) # computes the max pooling of the correct slices
        return pool_z   
    def conv_forward(self, z, K, b, padding=0):
       
       stride = 1 # forcing stride to be 1
       padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0) # line that automatically pads the function
       (N,C,H,W) = padding_z.shape # the following gathers the dimensions for the for loops
       (D,C,k1,k2) = K.shape 
       (OC,OD,OH,OW) = z.shape

       oH = int((OH+2*padding-k1)/stride)+1 # Scalling of the height
       oW = int((OW+2*padding-k2)/stride)+1 # scalling of the width for the ooutput matix

       conv_z = np.zeros([N,D,oH,oW]) # since appending numpy arrays are not computationally efficient, this makes an array of zeros of the right size that can be updated later
       for i in range (N): # iterates through the input images
          for j in range (D): # iterates through the filters
              # the rest goes through the height and width of the input images
              for h in range (oH): 
                  for w in range (oW):
                      conv_z[i,j,h,w] = np.sum(K[j,:] *padding_z[i,:,h:h+k1,w:w+k2])+b[j] # this code essentially takes the dot product of the input matrix section and the filter
       return conv_z
    def forwrd(self,x,conv1_w,conv1_b,conv2_w,conv2_b,fc_w,fc_b):

        conv1       =   self.conv_forward(x.astype(np.float64),conv1_w,conv1_b,padding=1)
        conv1_relu  =   self.relu_forward(conv1)
        maxp1       =   self.max_pooling_forward(conv1_relu.astype(np.float64))

        conv2       =   self.conv_forward(maxp1, conv2_w, conv2_b, padding=1)
        conv2_relu  =   self.relu_forward(conv2)
        maxp2       =   self.max_pooling_forward(conv2_relu.astype(np.float64))

        flatten     =   self.flatten_forward(maxp2)
        y           =   self.fc_forward(flatten,fc_w,fc_b)

        return y
                       
    pass




