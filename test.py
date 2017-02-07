#  -*- coding:utf-8 -*-
import numpy as np
import theano.tensor as T
from theano import function
import theano
from theano import pp
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
import pickle

# # basic
# x=T.dscalar('x')
# y=T.dscalar('y')
# z=x+y
# f=function([x,y],z)
# print (f(2,3))
#
# # to pretty-print the function
# print (pp(z))
#
# # how about matrix
# x=T.dmatrix('x')    #dmatrix代表64位，fmatrix代表32位
# y=T.dmatrix('y')
# z=x+y
# f=function([x,y],z)
# print (f(np.arange(12).reshape(3,4),10*np.ones((3,4))))

# activation function example
# x=T.dmatrix('x')
# s=1/(1+T.exp(-x))   # logistic or soft step
# logistic=theano.function([x],s)
# print (logistic([[0,1],[-2,-3]]))
# # print pp(s)

# multiply outputs for a function
# a,b=T.dmatrices('a','b')
# diff=a-b
# abs_diff=abs(diff)
# diff_squared=diff**2
# f=theano.function([a,b],[diff,abs_diff,diff_squared])
# print f(np.ones((2,2)),np.arange(4).reshape((2,2)))

#name for a function
# x,y,w=T.dscalars('x','y','w')
# z=(x+y)*w
# f=theano.function([x,theano.In(y,value=1),theano.In(w,value=2,name='weights')],z)
# print f(23,)
# print f(23,2,weights=4)

# share 变量
# state =theano.shared(np.array(0,dtype=np.float64),'state')
# inc= T.scalar('inc',dtype=state.dtype)
# accumulator=theano.function([inc],state,updates=[(state,state+inc)])
# print accumulator(10)
# print accumulator(10)
# print accumulator(10)
# to get variable value
# print state.get_value()
# accumulator(1)
# print state.get_value()
# accumulator(10)
# print state.get_value()
# # to set variable value
# state.set_value(-1)
# accumulator(3)
# print state.get_value()
# # teporarily replace shared variable with another value in another function
# tmp_func =state*2+inc
# a=T.scalar(dtype=state.dtype)
# skip_shared=theano.function([inc,a],tmp_func,givens=[(state,a)])
# print skip_shared(2,3)
# print state.get_value()

# # 回归拟合
# ##定义layer类
#
# class Layer(object):
#     def __init__(self,inputs,in_size,out_size,activation_function=None):
#         self.W= theano.shared(np.random.normal(0,1,(in_size,out_size)))
#         self.b= theano.shared(np.zeros((out_size,))+0.1)
#         self.Wx_plus_b= T.dot(inputs,self.W)+self.b
#         self.activation_function=activation_function
#         if activation_function is None:
#             self.outputs= self.Wx_plus_b
#         else:
#             self.outputs= self.activation_function(self.Wx_plus_b)
#
# """
# to define the layer like this:
# l1=Layer(inputs,in_size=1,out_size=10,activation_function)
# l2=Layer(l1.outputs,10,1,None)
# """
# #Make up some fake data
# x_data=np.linspace(-1,1,300)[:,np.newaxis]
# noise=np.random.normal(0,0.05,x_data.shape)
# y_data=np.square(x_data)-0.5+noise   #y=x**2-0.5
# #show the fake data
# # plt.scatter(x_data,y_data)
# # plt.show()
# #determine the inputs dtype
# x=T.dmatrix('x')
# y=T.dmatrix('y')
# #add layers
# l1=Layer(x,1,10,T.nnet.relu)
# l2=Layer(l1.outputs,10,1,None)
# #compute the cost
# cost=T.mean(T.square(l2.outputs-y))
# #compute the gradients
# gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])
# #apply gradient descent
# learing_rate=0.05
# train=theano.function(
#     inputs=[x,y],
#     outputs=cost,
#     updates=[(l1.W,l1.W-learing_rate*gW1),
#              (l1.b,l1.b-learing_rate*gb1),
#              (l2.W,l2.W-learing_rate*gW2),
#              (l2.b,l2.b-learing_rate*gb2)])
# #prediction
# predict=theano.function(inputs=[x],outputs=l2.outputs)
#
# #plot the fake data
# fig =plt.figure()
# ax= fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()
# for i in range(1000):
#     #training
#     err =train(x_data,y_data)
#     if i%50 ==0:
#         #to visualize the result and improvement
#         try:
#           ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         predicted_value=predict(x_data)
#         #plot the prediction
#         lines=ax.plot(x_data,predicted_value,'r-',lw=5)
#         plt.pause(1)
#         print err


# ##分类学习
# def compute_accuracy(y_target,y_predict):
#   correct_prediction =np.equal(y_predict,y_target)
#   accuracy=float(np.sum(correct_prediction))/len(correct_prediction)
#   return accuracy
# rng=np.random
# N=400    #training sample size
# feats=784  #number of input variable
#
# #generate a datasets: D=(input_values,target_calss)
# D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))
#
# #Declare Theano symbolic variables
# x=T.dmatrix('x')
# y=T.dvector('y')
#
# #initialize the weights and biases
# W=theano.shared(rng.randn(feats),name='w')
# b=theano.shared(0.1,name='b')
#
# #Construct Theano expression graph
# p_1 = T.nnet.sigmoid(T.dot(x,W)+b)
# prediction=p_1 >0.5
# xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1)
# cost=xent.mean()+0.01*(W**2).sum()  #l2
# gW,gb=T.grad(cost,[W,b])
#
# #Compile
# learing_rate=0.1
# train=theano.function(
#     inputs=[x,y],
#     outputs=[prediction,xent.mean()],
#     updates=((W,W-learing_rate*gW),
#              (b,b-learing_rate*gb)))
# #prediction
# predict=theano.function(inputs=[x],outputs=prediction)
# # training
# for i in range(500):
#     pred,err=train(D[0],D[1])
#     if i%50 ==0:
#       print 'cost:',err
#       print 'accuracy:',compute_accuracy(D[1],predict(D[0]))
# print "target values for D:"
# print D[1]
# print "prediction on D:"
# print predict(D[0])


# # #regularization l1,l2
# class Layer(object):
#     def __init__(self,inputs,in_size,out_size,activation_function=None):
#         self.W= theano.shared(np.random.normal(0,1,(in_size,out_size)))
#         self.b= theano.shared(np.zeros((out_size,))+0.1)
#         self.Wx_plus_b= T.dot(inputs,self.W)+self.b
#         self.activation_function=activation_function
#         if activation_function is None:
#             self.outputs= self.Wx_plus_b
#         else:
#             self.outputs= self.activation_function(self.Wx_plus_b)
#
# def minmax_normalization(data):
#     xs_max=np.max(data,axis=0)
#     xs_min=np.min(data,axis=0)
#     xs=(1-0)*(data-xs_min)/(xs_max-xs_min)+0
#     return xs
#
# np.random.seed(100)
# x_data=load_boston().data
# #minmax normalization,rescale the inputs
# x_data=minmax_normalization(x_data)
# y_data=load_boston().target[:,np.newaxis]
#
# #cross validation,train test data split
# x_train,y_train=x_data[:400],y_data[:400]
# x_test,y_test=x_data[400:],y_data[400:]
#
# x=T.dmatrix('x')
# y=T.dmatrix('y')
# #add layers
# l1=Layer(x,13,50,T.tanh)
# l2=Layer(l1.outputs,50,1,None)
# #compute the cost
# # cost=T.mean(T.square(l2.outputs-y))
# cost=T.mean(T.square(l2.outputs-y))+0.1*((l1.W**2).sum()+(l2.W**2).sum())  #l2 regularization
# # cost=T.mean(T.square(l2.outputs-y))+0.1*(abs(l1.W).sum()+abs(l2.W).sum())  #l1 regularization
# #compute the gradients
# gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])
# #apply gradient descent
# learing_rate=0.01
# train=theano.function(
#     inputs=[x,y],
#     outputs=cost,
#     updates=[(l1.W,l1.W-learing_rate*gW1),
#              (l1.b,l1.b-learing_rate*gb1),
#              (l2.W,l2.W-learing_rate*gW2),
#              (l2.b,l2.b-learing_rate*gb2)])
#
# compute_cost=theano.function(inputs=[x,y],outputs=cost)
#
# #record cost
# train_err_list=[]
# test_err_list=[]
# learning_time=[]
#
# for i in range(1000):
#     #training
#     train(x_train,y_train)
#     if i%10 ==0:
#         #record cost
#         train_err_list.append(compute_cost(x_train,y_train))
#         test_err_list.append(compute_cost(x_test, y_test))
#         learning_time.append(i)
# #plot cost history
# plt.plot(learning_time,train_err_list,'r-')
# plt.plot(learning_time,test_err_list,'b--')
# plt.show()

# ##保存与提取
# def compute_accuracy(y_target,y_predict):
#   correct_prediction =np.equal(y_predict,y_target)
#   accuracy=float(np.sum(correct_prediction))/len(correct_prediction)
#   return accuracy
# rng=np.random
# N=400    #training sample size
# feats=784  #number of input variable
#
# #generate a datasets: D=(input_values,target_calss)
# D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))
#
# #Declare Theano symbolic variables
# x=T.dmatrix('x')
# y=T.dvector('y')
#
# #initialize the weights and biases
# W=theano.shared(rng.randn(feats),name='w')
# b=theano.shared(0.1,name='b')
#
# #Construct Theano expression graph
# p_1 = T.nnet.sigmoid(T.dot(x,W)+b)
# prediction=p_1 >0.5
# xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1)
# cost=xent.mean()+0.01*(W**2).sum()  #l2
# gW,gb=T.grad(cost,[W,b])
#
# #Compile
# learing_rate=0.1
# train=theano.function(
#     inputs=[x,y],
#     outputs=[prediction,xent.mean()],
#     updates=((W,W-learing_rate*gW),
#              (b,b-learing_rate*gb)))
# #prediction
# predict=theano.function(inputs=[x],outputs=prediction)
# # training
# for i in range(500):
#     pred,err=train(D[0],D[1])
#
# #save model
# with open ('save/model.pickle','wb') as file:
#     model=[W.get_value(),b.get_value()]
#     pickle.dump(model,file)
#     print model[0][:10]
#     print "accuracy:",compute_accuracy(D[1],predict(D[0]))



#load model
def compute_accuracy(y_target,y_predict):
  correct_prediction =np.equal(y_predict,y_target)
  accuracy=float(np.sum(correct_prediction))/len(correct_prediction)
  return accuracy
rng=np.random
N=400    #training sample size
feats=784  #number of input variable

#generate a datasets: D=(input_values,target_calss)
D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))

#Declare Theano symbolic variables
x=T.dmatrix('x')
y=T.dvector('y')

#initialize the weights and biases
W=theano.shared(rng.randn(feats),name='w')
b=theano.shared(0.1,name='b')

#Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x,W)+b)
prediction=p_1 >0.5
xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1)
cost=xent.mean()+0.01*(W**2).sum()  #l2
gW,gb=T.grad(cost,[W,b])

#Compile
learing_rate=0.1
train=theano.function(
    inputs=[x,y],
    outputs=[prediction,xent.mean()],
    updates=((W,W-learing_rate*gW),
             (b,b-learing_rate*gb)))
predict=theano.function(inputs=[x],outputs=prediction)
with open ('save/model.pickle','rb') as file:
    model=pickle.load(file)
    W.set_value(model[0])
    b.set_value(model[1])
    print W.get_value()[:10]
    print "accuracy:",compute_accuracy(D[1],predict(D[0]))