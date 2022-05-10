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
    return f_w


def gradient_softmax(x, w, c):
    # print("shape c",c.shape)
    m_dim2 = x.shape[1]  # could be the batch size
    # print(x.shape)
    # print(w.shape)
    # print(c.shape)
    # print((np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c))
    # print("before before you yay here", np.subtract(np.divide(np.exp(x.T @ w), np.sum(np.exp(x.T @ w), axis=1).reshape(m_dim2, 1)),c))
    # print("yay",(x@(np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c)))
    soft_grad = (x @ (np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c)) \
                / x.shape[1]
    return soft_grad


def relu(x):
    return np.maximum(x, 0)


def grad_relu(x):
    return (x > 0) * 1
