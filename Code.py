import gzip,pickle,sys
import matplotlib.pyplot as plt
import numpy as np

conv1_w = np.load("conv1_w.np")
conv1_b = np.load("conv1_b.npy")
conv2_w = np.load("conv2_w.npy")
fc_w = np.load("fc_w.npy")
fc_b = np.load("fc_b.npy")

f = gzip.open('mnist.pkl.gz','rb')
if sys.version_info < (3,):
    (X_train, y_train),(X_test, y_test) = pickle.load(f)
else:
    (X_train, y_train), (X_test, y_test) = pickle.load(f, encoding="bytes")

idx=np.random.choice(X_test.shape[0],5)
x,y=X_test[idx],y_test[idx]
x = np.reshape(x,(-1,1,28,28))


for i in range(5):
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(x[i],(28,28)), cmap='gray')
    plt.show()
    print("y_true:{},y_predict:{}".format(y[i],np.argmax(y_predict[:,i])))