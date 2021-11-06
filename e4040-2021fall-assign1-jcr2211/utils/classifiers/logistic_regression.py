import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################

    h =  1/(1+np.exp(-x))

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0.0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    N = X.shape[0]
    h = (sigmoid(np.dot(X,W)))
    y = y.reshape(N,1)
    for hi,yi in zip(h,y):
      loss += (yi*np.log(hi) + (1-yi)*np.log(1-hi))/N
    loss += reg*np.linalg.norm(W)**2
    #dW = np.dot(X.T, (h - y).T) / y.shape[0]
    for i,dWi in enumerate(dW):
      dW[i] = np.mean(np.dot(X.T[i], (h - y)))
    #dW = np.dot(X.T, (h - y)).mean(axis=1)
    dW = dW.reshape(len(dW),1) + 2*reg*np.linalg.norm(W)**2
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss[0],dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    #print("W shape",W.shape)
    h = (sigmoid(np.dot(X,W)))
    y = y.reshape(len(y),1)
    loss = (y * np.log(h) + (1 - y) * np.log(1 - h)).mean() + reg*np.linalg.norm(W)**2
    #dW = np.dot(X.T, (h - y).T) / y.shape[0]
    dW = np.dot(X.T, (h - y)).mean(axis=1) + 2*reg*np.linalg.norm(W)**2
    dW = dW.reshape(len(dW),1)

    #print("dW shape",dW.shape)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW
