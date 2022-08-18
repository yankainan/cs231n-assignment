import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, C = X.shape[0], W.shape[1]
  for i in range(N):
    #* 计算loss
    f = np.dot(X[i], W) #! 计算这个样本属于各个类别的概率
    f -= np.max(f) #! 这里注意为了数值稳定还要加上 -np.max(f)
    loss += np.log(np.sum(np.exp(f))) - f[y[i]]
    #* 计算dW
    dW[:,y[i]] -= X[i] #!对损失函数求导可知
    s = np.sum(np.exp(f))
    for j in range(C):
      dW[:,j] += np.exp(f[j]) * X[i] / s
      
    #* 平均并添加正则化项
  loss = loss / N + reg * np.sum(W * W)
  dW = dW / N + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, C = X.shape[0],W.shape[1]
  #* 计算 loss
  f = np.dot(X,W) 
  f = f - np.max(f, 1).reshape(N,1)#! 注意这里是减去每一个样本中得分最高的那个类，而不是所有样本得分最高的那个类
  s = np.sum(np.exp(f),1)
  loss = np.sum(np.log(s) - f[range(N),y])
  
  #* 计算 dW
  temp = np.exp(f) / s.reshape(N,1)
  temp[range(N), y] -= 1
  dW = np.dot(X.T, temp)
  
  #* 加入正则项
  loss = loss / N + reg * np.sum(W * W)
  dW = dW / N + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

