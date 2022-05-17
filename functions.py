import numpy as np


def softmax_func(x, w, c):  # loss func!
    """
    softmax_func is

    :param x: X - Matrix with n x m dimensions (n pixels of each image, m images dataset)
    :param c: C - Matrix is a diagonal matrix with m x l (l different classes)
    :param w: W - Matrix is a weights' matrix with dimensions n x l
    :return:
    """
    n_dim, m_dim = x.shape
    l_dim = c.shape[1]
    # print("scipy softmax", softmax(x.T @ w))
    f_w = np.sum(
        -1 * np.multiply(c, np.log(np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim, 1))))))
    return f_w  # the -1 is necessery?


def gradient_softmax(x, theta, c, wrt="w"):
    w, b = theta
    m_dim2 = x.shape[1]
    if wrt == 'w':
        # print("shape c",c.shape)
        # print(x.shape)
        # print(        # print((np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c))c.shape)
        # print("before before you yay here", np.subtract(np.divide(np.exp(x.T @ w), np.sum(np.exp(x.T @ w), axis=1).reshape(m_dim2, 1)),c))
        # print("yay",(x@(np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c)))
        soft_grad = (x @ (
                    np.divide(np.exp(x.T @ w + b), (np.sum(np.exp(x.T @ w + b), axis=1).reshape(m_dim2, 1))) - c)) \
                    / x.shape[1]
        print("w soft",soft_grad)
        print("w dimensions",soft_grad)
        return soft_grad
    elif wrt == 'x':
        print("Expected X.T shape (3,10), got: ",x.T.shape)
        print("Expected W shape (10,2), got:",w.shape)
        print("Expected shape (3,1), got:",np.sum(np.exp(x.T @ w + b),axis=1).shape)

        print("Expected C shape (3,2)",c.shape)
        print("shit",(np.divide(np.exp(x.T @ w + b), (np.sum(np.exp(x.T @ w + b), axis=1).reshape((m_dim2, 1))))))
        soft_grad = (w @ (
                    np.divide(np.exp(x.T @ w + b), (np.sum(np.exp(x.T @ w + b), axis=1)).reshape((m_dim2, 1)) - c).T) \
                        / x.shape[1])
        print("expected soft_rade (10,3) got:",soft_grad.shape)
        return soft_grad
    else:  # bias
        soft_grad = (np.divide(np.exp(x.T @ w + b),
                               (np.sum(np.exp(np.add(x.T @ w, b)), axis=1)).reshape((m_dim2, 1))) - c) \
                    / x.shape[1]
        soft_grad = np.sum(soft_grad,axis=0).T
        return soft_grad


def relu(x):
    print("in relu before",x.shape)
    x = np.maximum(x,0)
    print("in reu after",x.shape)

    return np.maximum(x, 0)


def grad_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
