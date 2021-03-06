import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax


# def softmax(x):
#     # z--> linear part.
#
#     # subtracting the max of z for numerical stability.
#
#     sum_of_exps = np.sum(np.exp(x))
#
#     # Calculating softmax for all examples.
#     for i in range(len(z)):
#         exp_xt_w = np.sum(exp)
#         exp[i] /= np.sum(exp[i])
#
#     return exp
def test_linear_regression():
    data = np.array([[1, 1], [2, 2], [4, 3], [6, 5], [1, 2]]).T
    print(data.shape[1])
    new_w2 = np.random.rand(2)
    for epoch in range(1000):
        new_w2 = loss_func_SGD(gradient_linear, data, new_w2, mini_batch=1)
    print(new_w2)
    m, b = new_w2
    print(data)
    x = data[0, :]
    y = data[1, :]
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [min(m * x + b), max(m * x + b)], color='green')  # regression line
    plt.show()

def softmax_func(x, c, w):
    """
    softmax_func is

    :param x: X - Matrix with n x m dimensions (n pixels of each image, m images dataset)
    :param c: C - Matrix is a diagonal matrix with m x l (l different classes)
    :param w: W - Matrix is a weights' matrix with dimensions n x l
    :return:
    """
    n_dim, m_dim = x.shape
    print(n_dim, m_dim)
    l_dim = c.shape[1]
    print(l_dim)
    print("scipy softmax", softmax(x.T @ w))
    f_w = np.sum(
        np.multiply(c, np.log(np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim, 1))))))
    print(f_w)


def gradient_softmax(x, w, c):
    print("shape c",c.shape)
    m_dim2 = x.shape[1] # could be the batch size
    print("before before you yay here", np.subtract(np.divide(np.exp(x.T @ w), np.sum(np.exp(x.T @ w), axis=1).reshape(m_dim2, 1)),c))
    print("yay",(x@(np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c)))
    soft_grad = (x @ (np.divide(np.exp(x.T @ w), (np.sum(np.exp(x.T @ w), axis=1)).reshape((m_dim2, 1))) - c)) \
                / x.shape[1]
    return soft_grad


def gradient_linear(x, y, w):
    """
    gradient_linear is a function with a close formula for the gradient of the residual errors (D_m,D_b) when
                m is the slope and b is the bias/offset from the x-axis.
                X=x1|x2
                   x1 = (a0,b0)
    :param x:
    :param y:
    :param w:
    :return:
    """
    samples = x.shape[0]  # the number of samples
    m, b = w
    y_pred = m * x + b
    d_m = (-2 / samples) * np.sum(x @ (y - y_pred))  # Derivative wrt m
    d_b = (-2 / samples) * np.sum(y - y_pred)  # Derivative wrt c

    return np.array([d_m, d_b])


def loss_func_SGD(loss_grad, x, w, c=None, mini_batch=4, learning_rate=0.0001):
    """
    loss_func_SGD is a function that finding the minimum of the loss function, given the gradient of the loss function
    and the current weight. The fuction is finding the minimum using Stochastic Gradient Decent method
    that updates the W that minimize the loss function for the current mini batch images.
    :param loss_grad: The close formula of the gradient of the loss function.
    :param x: The data X=[x1|x2|x3...|xn], dimensions: n x m
    :param w: The weights, the parameters that we try to find. If is a linear regression then w=[m,b] when y=mx+b
    :param c: The matrix C=[c1|c2|c3|...|cn] when ci is a {0,1}^n vector, when every entry j is 1 if the ground truth of
                image xj is class i and 0 otherwise.
    :param mini_batch: The amount of images we want to use to update the weights for the Stochastic Gradient Decent
                        method.
    :param learning_rate: The
    :return:
    """

    if c is None:  # Linear Regression
        new_w = w.copy()
        iteration = 1
        # if mini_batch == 1:
        #     batch_begin = 0
        #     batch_end = 1
        batch_begin = iteration * mini_batch - mini_batch
        batch_end = (iteration * mini_batch)
        while batch_end <= x.shape[1]:  # update the weights for one epoch
            # new_m = m - learning_rate*f_m
            # new_b = b - learning_rate*f_b
            # new_w = [m - learning_rate*f_m, b - learning_rate*f_b]
            new_w = new_w - learning_rate * gradient_linear(x[0, batch_begin:batch_end], x[1, batch_begin:batch_end],
                                                            new_w)
            iteration += 1
            batch_begin = iteration * mini_batch - mini_batch + 1
            batch_end = (iteration * mini_batch) + 1
        # update the weights for the last data if it wasn't used in the iterations because of the batch size.
        new_w = new_w - learning_rate * gradient_linear(x[0, x.shape[1] - batch_end: x.shape[1]],
                                                        x[1, x.shape[1] - batch_end: x.shape[1]], new_w)
        return new_w
    else:  # Logistic Regression
        new_w = w.copy()
        iteration = 1
        batch_begin = iteration * mini_batch - mini_batch
        batch_end = (iteration * mini_batch)
        while batch_end <= x.shape[1]:  # until we still got enough images in the current epoch for the mini-batch
            new_w = learning_rate * loss_grad(x[:, batch_begin:batch_end], new_w, c)
            iteration += 1
            batch_begin = iteration * mini_batch - mini_batch + 1
            batch_end = (iteration * mini_batch) + 1
        # calculate the rest of the images when the number of them is less than the mini batch
        new_w = learning_rate * loss_grad(x[:, x.shape[1] - batch_end: x.shape[1]], new_w, c)
    return new_w


if __name__ == "__main__":
    # example of pictures
    # mat0 = np.arange(12).reshape(4, 3).flatten()
    # mat1 = np.arange(12).reshape(4, 3).flatten()
    # mat2 = np.arange(12).reshape(4, 3).flatten()
    # mat3 = np.arange(12).reshape(4, 3).flatten()

    mat0 = np.random.rand(4,3).flatten()
    mat1 = np.random.rand(4,3).flatten()
    mat2 = np.random.rand(4,3).flatten()
    mat3 = np.random.rand(4,3).flatten()

    # X = [x1|x2|x3..]
    given_x = np.array([mat0, mat1, mat2, mat3]).T

    given_c = np.matrix([[1, 0, 0, 0], [0, 1, 1, 1]]).T
    n_dim, m_dim = given_x.shape
    l_dim = given_c.shape[1]

    rand_w = np.random.rand(n_dim, l_dim)

    print(given_c)
    print(given_x)
    softmax_func(given_x, given_c, rand_w)
    # print(gradient_softmax(given_x, rand_w, given_c))
    # print(np.sum(c,axis=1))
    # print(np.divide(mat,np.sum(c,axis=1)))
    print("What before")
    weights=rand_w
    for epoch in range(1000):
        weights = loss_func_SGD(gradient_softmax,given_x,weights,given_c,mini_batch=2)
    print("weights",weights)
    print("check sanity",mat1.T*weights)
    # test_linear_regression()
    # new_w2 = rand_w
    # for epoch in range(10):
    #     new_w2 = loss_func_SGD(gradient_linear, data, new_w2, mini_batch=1)
    # print(new_w2)
    # m,b = new_w2
    # print(data)
    # x = data[0, :]
    # y = data[1,:]
    # print(x)
    # print(y)
    # plt.scatter(x, y)
    # plt.plot([min(x), max(x)], [min(m * x + b), max(m * x + b)], color='green')  # regression line
    # plt.show()
