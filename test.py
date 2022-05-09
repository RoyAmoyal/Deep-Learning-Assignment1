import functions as fn
import numpy as np


def random_x(n, m):
    mat0 = np.random.rand(n,m).flatten() # 'pictures' for example
    mat1 = np.random.rand(n,m).flatten()
    mat2 = np.random.rand(n,m).flatten()
    mat3 = np.random.rand(n,m).flatten()
    return np.array([mat0, mat1, mat2, mat3]).T

def x_w_c():
    x = random_x(4, 3)
    c = np.matrix([[1, 0, 0, 0], [0, 1, 1, 1]]).T
    n_dim, m_dim = x.shape  # 12 and 4
    l_dim = c.shape[1]      # 2
    w = np.random.rand(n_dim, l_dim)
    return x, w, c

def softmax_grad_test():
    softmax = fn.softmax_func
    softmax_grad = fn.gradient_softmax
    epsilon = 0.1
    x, w, c = x_w_c()
    # x_noise = np.random.rand(x.shape)
    w_noise = np.random.rand(w.shape[0], w.shape[1])
    f0 = softmax(x, w, c)
    g0 = softmax_grad(x, w, c)
    for i in range(8):
        epsk = epsilon*((0.5)**(i+1))
        fk = softmax(x, w + epsk*w_noise, c)
        f1 = f0 + epsk*w_noise.T*softmax_grad(x, w, c)
        print(i+1, "\t", abs(fk - f0), "\t", abs(fk - f1))

softmax_grad_test()





