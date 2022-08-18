from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        hidden_layer, hidden_layer_cache = affine_relu_forward(X, W1, b1)
        scores, scores_cache = affine_forward(hidden_layer, W2, b2) #! 注意这层输出分数的不需要使用ReLU激活函数
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscore = softmax_loss(scores, y)
        #! 注意计算时传入的每一层的参数
        dhidden_layer, grads['W2'], grads['b2'] = affine_backward(dscore, scores_cache) #! 输出层是不需要ReLU的，因此只需要简单的反向传播即可
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dhidden_layer, hidden_layer_cache)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2)) #! 加上正则项，注意提示说的使用0.5进行简化梯度表达式
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #* 至少会有一层隐藏层，所以先初始化一下
        self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale #! 注意上面题目的说法，权重初始化的标准差应该为 weight_scale
        self.params['b1'] = np.zeros(hidden_dims[0])
        for i in range(self.num_layers - 2): #! 因为输出的那层不算，第一层已经初始化过了，所以是减去2
          #* 初始化第二个隐藏层开始的层
          self.params['W' + str(i + 2)] = np.random.randn(hidden_dims[i], hidden_dims[i + 1]) * weight_scale
          self.params['b' + str(i + 2)] = np.zeros(hidden_dims[i + 1])
          if self.use_batchnorm:
            self.params['gamma' + str(i + 1)] = np.ones(hidden_dims[i])
            self.params['beta' + str(i + 1)] = np.zeros(hidden_dims[i])
        
        #! 如果用了BN，仔细观察上面的循环，从倒数第二层到输出的那层还没有初始化BN，所以下面这步是用来初始化最后的BN的
        if self.use_batchnorm:
          self.params['gamma' + str(self.num_layers - 1)] = np.ones(hidden_dims[-1])
          self.params['beta' + str(self.num_layers - 1)] = np.zeros(hidden_dims[-1])
          
        #! 初始化输出层的参数
        self.params['W' + str(self.num_layers)] = np.random.randn(hidden_dims[-1], num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        netlayers, cache = list(range(self.num_layers + 1)), list(range(self.num_layers)) #! netlayers包含了所有网络层的信息，第0层就是输入层，最后一层是输出层，#! 注意这里的cache是存储的前一层到该层的参数，所以只用self.num_layers就可以，下标到self.num_layers - 1就够
        netlayers[0] = X #! 注意下标为0的要初始化为输入X
        dropout_cache = list(range(self.num_layers - 1)) #! 由于有可能有用了BN又用了dropout的情况，所以要单独设一个dropout的cache
        for i in range(self.num_layers):
          W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
          if i == self.num_layers - 1: #! 注意这里，如果是最后一层网络也就是得分输出的层，就不需要使用dropout或者BN了
            netlayers[i + 1], cache[i] = affine_forward(netlayers[i], W, b)
          else:
            if self.use_batchnorm:#! 判断有没有用BN，有的话就先bn再relu，没有的话就直接relu输出
              gamma, beta = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)]
              netlayers[i + 1], cache[i] = affine_bn_relu_forward(netlayers[i], W, b, gamma, beta, self.bn_params[i])#! 使用use_bn函数来计算并批量归一化输出
            else:
              netlayers[i + 1], cache[i] = affine_relu_forward(netlayers[i], W, b)
            if self.use_dropout:
              netlayers[i + 1], dropout_cache[i] = dropout_forward(netlayers[i + 1], self.dropout_param)
              
          scores = netlayers[-1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscore = softmax_loss(scores, y)
        dhidden = list(range(self.num_layers + 1))
        dhidden[self.num_layers] = dscore
        for i in range(self.num_layers, 0, -1):#! 可以用这个技巧更直观的表示反向传播
          if i == self.num_layers:
            dhidden[i - 1], grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dhidden[i], cache[i - 1])#! 这里对于下标可能会混淆，这个循环是以输出层为基础（所以一开始 i 是输出层的下标）从倒数第二层开始计算梯度，因此传入backward的应该是后一层（以输入层为前，输出层为后）的输入的梯度dhidden[i]和该层传到后一层的参数所以是cache[i - 1]，而传出的参数第一个是dhidden[i - 1]是 i - 1 层输出（i 层输入）的梯度，在前向传播中输入层是第0层，因此第 i 层使用的是 {W/b}_i + 1 的下标
          else:
            if self.use_dropout: #! 注意前向传播的时候是先BN再dropout，所以反向传播的时候先计算dropout再BN
              dhidden[i] = dropout_backward(dhidden[i], dropout_cache[i - 1]) #! 注意前向传播时是对第 i 层进行drop得到一个结果，所以得到的还是dhidden[i]
            if self.use_batchnorm:
              dhidden[i - 1], grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)], = affine_bn_relu_backward(dhidden[i], cache[i - 1])
            else: #! 现在是BN和dropout都不用的情况，但还是要注意是有ReLU的
              dhidden[i - 1], grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(dhidden[i], cache[i - 1])             
          loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] ** 2) #! 根据提示这里的正则化要策划功能上0.5来简化
          grads['W' + str(i)] += self.reg * self.params['W' + str(i)] #! 注意参数 W_i 的梯度要加上 reg * W_i ，因为为了正则化 W ，每一层的损失函数都有对于 W 的正则化项 reg * np.sum(W_i * W_i)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
