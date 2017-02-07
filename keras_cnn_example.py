#encoding=utf-8
#CPU或GPU测试
# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time
#
# vlen = 10 * 30 * 768
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in xrange(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')


import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape
from matplotlib import pyplot as plt
plt.imshow(X_train[0])












