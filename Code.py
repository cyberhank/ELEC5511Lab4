import gzip,pickle,sys
import matplotlib.pyplot as plt
import numpy as np
from  Network import NeuralNetwork
import pandas as pd

convy = NeuralNetwork() # enstantiates NeuralNetwork object

# Loads files
conv1_w = np.load("conv1_w.npy") 
conv1_b = np.load("conv1_b.npy")
conv2_w = np.load("conv2_w.npy")
conv2_b = np.load("conv2_b.npy")
fc_w = np.load("fc_w.npy")
fc_b = np.load("fc_b.npy")

# Unpickleing of initial files
f = gzip.open('mnist.pkl.gz','rb')
if sys.version_info < (3,):
    (X_train, y_train),(X_test, y_test) = pickle.load(f)
else:
    (X_train, y_train), (X_test, y_test) = pickle.load(f, encoding="bytes")

#Random pick of five starter files
idx=np.random.choice(X_test.shape[0],5)
x,y=X_test[idx],y_test[idx]
x = np.reshape(x,(-1,1,28,28))

#running of the code
y = convy.forwrd(x,conv1_w,conv1_b,conv2_w,conv2_b,fc_w,fc_b)
print(y)

#Displaying of the results
for i in range(5):
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(x[i],(28,28)), cmap='gray')
    plt.show()
    print("y_true:{},y_predict:{}".format(y[i],np.argmax(y[:,i])))