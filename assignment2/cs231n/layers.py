from builtins import range
from random import sample
from re import L
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N, D, M = x.shape[0], w.shape[0], b.size
    out = np.dot(x.reshape(N, D), w) + b.reshape(1, M)#! 将数据转化为（数据i， 数据i的特征），即将特征展开为一维
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):#! 注意这里的cache是存储的前一层到该层的参数
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N, D = x.shape[0], w.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape) #! 注意dout和实际的dx的维度是不同的，在这次作业中dx实际维度是(10, 2, 3)，而dout是(10, 6)，因此注意计算dx的时候要恢复维度和 x 一样
    dw = np.dot(x.reshape(N, D).T, dout)
    db = np.sum(dout, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db #! 返回的第一个参数就是这一网络层的输入，将其返回到上层进行反向传播


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']  #* 确定是训练的还是测试时候的参数
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    #! running_mean和running_var用于测试数据的时候使用
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.mean(x, 0)
        sample_var = np.var(x, 0)
        #* 对输入的数据x进行BN处理
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        cache = (x, gamma, beta, x_norm, sample_mean, sample_var, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    x, gamma, beta, x_norm, sample_mean, sample_var, eps = cache
    N, D = x.shape[0], x.shape[1]
    #! 计算dx——————————————https://whu-pzhang.github.io/cs231n-assignment2/
    #! 上面计算BN的步骤可以分为下面8步，其中要注意的是求均值和方差也涉及了数据x，因此求dx不能掉以轻心，要一步步倒推上去
    #* mean = 1.0 / N * np.sum(x, axis=0, keepdims=True)               # (1)
    #* xsubmean = x - mean                                             # (2)
    #* xsubmeansqr = xsubmean**2                                       # (3)
    #* var = 1.0 / N * np.sum(xsubmeansqr, axis=0, keepdims=True)      # (4)
    #* sqrtvar = np.sqrt(var + eps)                                    # (5)  #!注意这里的要加eps
    #* invsqrtvar = 1.0 / sqrtvar                                      # (6)
    #* x_norm = xsubmean * invsqrtvar                                  # (7)
    #* out = gamma * x_norm + beta                                     # (8)
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(x_norm * dout, 0)
    dx_norm = dout * gamma
    sqrtvar = np.sqrt(sample_var + eps)
    invsqrtvar = 1.0 / sqrtvar
    dxsubmean = dx_norm * invsqrtvar
    xsubmean = x_norm * sqrtvar  #! x减去均值后的结果，是利用了标准化的数据计算出来的
    dinvsqrtvar = np.sum(dx_norm * xsubmean, 0) 
    dsqrtvar = dinvsqrtvar * (-1.0) * sqrtvar**(-2)
    dvar = dsqrtvar * 0.5 * (sample_var + eps)**(-0.5)
    dxsubmeansqr = dvar * 1.0 / N * np.ones((N, D)) #! 注意这里有个用全一矩阵扩充样本方差的关键步骤
    dxsubmean += dxsubmeansqr * 2 * xsubmean
    dx = dxsubmean
    dmean = -1.0 * np.sum(dxsubmean, 0)
    dx += dmean * 1.0 / N * np.ones((N, D))  #! 注意这里是用的+=，因为mean的计算也用到了x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, gamma, beta, x_norm, sample_mean, sample_var, eps = cache
    N, D = dout.shape[0], dout.shape[1]
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(dout * x_norm, 0)
    #! 对于x的求导：多元复合函数的求偏导法则:多元复合函数对某一自变量的偏导数，等于这个函数对各个中间变量的偏导数与这个中间变量对该自变量的偏导数的乘积和————锁链法则或链法则。
    #! 证明————https://zhuanlan.zhihu.com/p/358620581（没看明白，记住链式求导可以这么算就好，会用就行）
    #! 这部分的计算总结可以看https://whu-pzhang.github.io/cs231n-assignment2/
    #* var_eps = sample_var + eps
    #* dx = (gamma * var_eps**(-0.5) / N) * (N * dout - np.sum(dout * x_norm, 0) * x_norm - np.sum(dout, 0))
    #* 上面这行是使用https://whu-pzhang.github.io/cs231n-assignment2/的最后结果实现的，下面的实现更通俗，使用链式法则
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * -0.5 * (x - sample_mean) * (sample_var + eps)**(-1.5), axis = 0)
    dmean = np.sum((-1) * dx_norm * (sample_var + eps)**(-0.5), axis=0) + np.sum(dvar * 2 / N * (x - sample_mean), axis=0) #! 这里也用了多元复合函数求偏导的链式法则
    dx = dx_norm * (sample_var + eps)**(-0.5) + dvar*2*(x - sample_mean) / N + dmean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p  #! p是比如一个神经元的输出是x，那么在训练的时候它有p的概率参与训练，(1-p)的概率丢弃
        #! 注意传入维度时需要加上*号
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x  #! 本来是应该乘一个drop的概率（即参与训练的概率）p的，但是在训练的时候已经先除去了，现在再乘上去相当于乘个1，就可以忽略这一步
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad, stride = conv_param['pad'], conv_param['stride']
    #* 生成扩展pad列后的x
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant') #! 注意扩充时的函数写法
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape  #! 这个维度F可能有点难理解，我觉得可以理解为有F个卷积核
    output_H = int(1 + (H + 2 * pad - HH) / stride)
    output_W = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N, F, output_H, output_W)) #! 由于一开始out是None，所以要先初始化一下，每个样本从（C， H， W）变成（F， HH， WW）
    for n in range(N):
        for f in range(F):
            for i in range(output_H):
                for j in range(output_W):
                    out[n, f, i, j] = np.sum(x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] * w[f, :, :, :]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #* 接受传进来的参数
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    output_H = int(1 + (H + 2 * pad - HH) / stride)
    output_W = int(1 + (W + 2 * pad - WW) / stride)

    #* 初始化梯度
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant')
    dx = np.zeros(x.shape)
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    #* x: Input data of shape (N, C, H, W)
    #* w: Filter weights of shape (F, C, HH, WW)
    #* b: Biases, of shape (F,)
    
    #* 开始计算反向传播梯度
    for n in range(N):
        for f in range(F):
            for i in range(output_H):
                for j in range(output_W):
                    x_pad_mask = x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    db[f] += dout[n, f, i, j]
                    dw[f, :, :, :] += dout[n, f, i, j] * x_pad_mask
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += dout[n, f, i, j] * w[f, :, :, :]
    
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    output_height = int(H / pool_height)
    output_weight = int(W / pool_width)

    out = np.zeros((N, C, output_height, output_weight))  #! 注意这里要初始化

    #* 开始计算池化层结果
    for i in range(output_height):
        for j in range(output_weight):
            x_mask = x[:, :, i * stride:i * stride + pool_height,
                       j * stride:j * stride + pool_width]
            out[:, :, i, j] = np.max(x_mask,
                                     axis=(2, 3))  #! 找出每一个数据在每一个C的层的最大值
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    output_height = int(H / pool_height)
    output_width = int(W / pool_width)
    dx = np.zeros(x.shape)

    for i in range(output_height):
        for j in range(output_width):
            x_mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            dx[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] += dout[:, :, i, j][:, :, None, None] * (x_mask == np.max(x_mask, axis=(2, 3), keepdims=True))  #! 反向传播时，只是最大值的那个点有梯度回传回去
            #! 注意这个keepdims=True，让我吃了苦头，这个参数的意思是保持维度特性
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #* spatial batch normalization computes a mean and variance for each of
    #* the C feature channels by computing statistics over both the minibatch
    #* dimension N and the spatial dimensions H and W.
    #! 注意上面原论文的陈述，意思是空间批量归一化通过计算小批量维度 N 和空间维度 H 和 W
    #! 的统计数据来计算每个 C 个特征通道的均值和方差。因此这 C 个特征通道可以看作是数据的
    #! 特征来转换
    N, C, H, W = x.shape
    #! 注意 x 压缩和 out 展开的顺序
    temp_out, cache = batchnorm_forward(x.transpose(0, 3, 2, 1).reshape(N * W * H, C), gamma, beta, bn_param)
    #! 上面这行代码首先使用transpose函数变化维度是因为由于我们调用了之前的BN_forward函数，它的输入 x 的维度是（N, D），我们把维度 C 的那一层当成了输入的 x 的 D 的那一层，但是在reshape的时候是按照数据从低维到高维，每一维从下标小到下标大来调整的，因此为了 C 维的那一层保持不变的放在输入 x 的特征 D 的那一层，需要先transpose再reshape
    out = temp_out.reshape(N, W, H, C).transpose(0, 3, 2, 1)  #! 转换output的维度为正常的输出维度
    #! 由于在transpose的时候是 N-W-H-C ，因此在reshape压缩的时候是 N-W-H-C 的顺序进行压缩的，所以上一行在展开的时候也要按照 N-W-H-C 的顺序展开
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0, 3, 2, 1).reshape(N * W * H, C), cache)
    dx = dx.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx #! 我认为 dx 可以看作是返回损失函数 softmax 对于输入的 x 的求导，也就是找出 softmax 在输入 x 上下降速度最快的方向
