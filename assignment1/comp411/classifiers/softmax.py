from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
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
    - regtype: Regularization type: L1 or L2

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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # pass
    
    matmultres = X.dot(W);
    classes_count = W.shape[1]
    training_data_count = X.shape[0]
    
    if regtype == 'L2':
        for i in range(training_data_count):
            matmultres_biased = matmultres[i] - np.max(matmultres[i]) # I found this in internet. In the towardsdatascience.com there is an example in which they do this and they say the reason is that they want to avoid very big numbers after exponentiation
            softmax = np.exp(matmultres_biased)/np.sum(np.exp(matmultres_biased))
            loss = loss -np.log(softmax[y[i]])
            
            for j in range(classes_count):
                dW[:,j] = dW[:,j] + X[i] * softmax[j]
                
            dW[:,y[i]] = dW[:,y[i]] - X[i]   
            
        loss = loss / training_data_count
        dW = dW / training_data_count
        
        loss = loss + reg * np.sum(W * W)
        dW = dW + 2 * reg * W 
        
    else:
        L1_gradient_term = np.ones_like(W)
        
        for i in range(training_data_count):
            matmultres_biased = matmultres[i] - np.max(matmultres[i]) # I found this in internet. In the towardsdatascience.com there is an example in which they do this and they say the reason is that they want to avoid very big numbers after exponentiation
            softmax = np.exp(matmultres_biased)/np.sum(np.exp(matmultres_biased))
            loss = loss -np.log(softmax[y[i]])
            
            for j in range(classes_count):
                dW[:,j] = dW[:,j] + X[i] * softmax[j]
                
            dW[:,y[i]] = dW[:,y[i]] - X[i]    
            
        loss = loss / training_data_count
        dW = dW / training_data_count
        
        loss = loss + reg * np.sum(abs(W))
        L1_gradient_term = np.where(W>0,L1_gradient_term,-1)
        dW = dW + reg * L1_gradient_term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X,  y, reg, regtype='L2'):
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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    training_data_count = X.shape[0]
    matmultres = X.dot(W)
    matmultres_biased = matmultres - np.max(matmultres, axis = 1, keepdims = True)
    
    if regtype == 'L2':
        exp__matmultres_biased = np.exp(matmultres_biased)
        sum_exp__matmultres_biased = exp__matmultres_biased.sum(axis = 1, keepdims=True)
        softmax_mat = exp__matmultres_biased/sum_exp__matmultres_biased
        
        loss = np.sum(-np.log(softmax_mat[np.arange(training_data_count), y]) )
        
        softmax_mat[np.arange(training_data_count),y] = softmax_mat[np.arange(training_data_count),y] - 1
        dW = X.T.dot(softmax_mat)
        
        loss /= training_data_count
        dW /= training_data_count
        
        loss = loss + reg * np.sum(W * W)
        dW = dW + 2 * reg * W
    else:
        L1_gradient_term = np.ones_like(W)
        
        exp__matmultres_biased = np.exp(matmultres_biased)
        sum_exp__matmultres_biased = exp__matmultres_biased.sum(axis = 1, keepdims=True)
        softmax_mat = exp__matmultres_biased/sum_exp__matmultres_biased
        
        loss = np.sum(-np.log(softmax_mat[np.arange(training_data_count), y]) )
        
        softmax_mat[np.arange(training_data_count),y] = softmax_mat[np.arange(training_data_count),y] - 1
        dW = X.T.dot(softmax_mat)
        
        loss /= training_data_count
        dW /= training_data_count
        
        loss = loss + reg * np.sum(abs(W))
        L1_gradient_term = np.where(W>0,L1_gradient_term,-1)
        
        dW = dW + reg * L1_gradient_term
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
