import numpy as np
import functions as F


class Model():
    def __init__(self, data, labels, minibatch=1, epochs=1,learning_rate=0.001):
        self.data = data
        self.labels = labels
        self.mini_batch = minibatch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.thetha_list = []
        self.grad_x_list = []
        self.x_list = []
        self.curr_c = []

    def forward(self, x):
        """
        :param x:
        :return:
        """
        self.x_list.append(x)
        curr_w = None
        for layer in self.thetha_list:
            curr_w = layer[0]
            curr_b = layer[1]
            x = np.add(curr_w @ x, curr_b)
            self.grad_x_list.append(np.multiply(curr_w, F.grad_relu(x)))
            self.x_list.append(x)
        self.grad_x_list.append(F.gradient_softmax(x, curr_w, self.labels))

        return F.softmax_func(x, curr_w, self.curr_c)

    def add__hidden_layer(self, m, entries, res=False):
        """
        Wx + b right now fix size
        :param n:
        :param m:
        :param res:
        :return:

        X=[x1|x2|x3]

        W mashu x n + b
        """
        # W*X W
        w = np.random.rand(m, entries)
        b = np.random.rand(m, 1)
        self.thetha_list.append((w, b))

    def backward(self):
        """
        fn = Wn*x+ bn
        :return:
        """
        jacob_gradients = []

        layer_num = len(self.grad_x_list) - 1

        last_x = self.x_list[layer_num]
        last_theta = self.thetha_list[layer_num]  # (w_last,b_last)
        dfn_theta = F.gradient_softmax(last_x, last_theta(0), c=self.curr_c)  # gradient of the loss function, last_theta(0) is w
        jacob_gradients.insert(0,dfn_theta)

        # update last layer
        v = self.grad_x_list[layer_num]
        for curr_theta in self.thetha_list[:len(self.thetha_list) - 2]:  # TODO check indexes
            layer_num -= 1
            curr_w, curr_b = curr_theta
            curr_x = self.x_list[layer_num]
            curr_df_theta = np.multiply(F.grad_relu(curr_x), v) @ curr_x.T  # w.r.t w, J * v
            jacob_gradients.insert(0, curr_df_theta)
            v = v @ self.grad_x_list[layer_num]  # add the next derivative to v for the backpropagation

        return jacob_gradients # the gradients in rows -df1-,-df2- and so on

    def update_theta(self,jacob_gradients):
        for index,grad in enumerate(jacob_gradients):
            self.thetha_list[index] = self.thetha_list[index] - self.learning_rate*grad


    # def resblock():
    #     pass

    def train(self):
        # loop over mini batches / epochs
        # TODO

        for epoch in range(self.epochs):
            self.forward(self.data)
            jacob_gradients = self.backward()
            self.update_theta(jacob_gradients) # updating the weights

            # clear x list and grad list and update weights
            self.grad_x_list = []
            self.x_list = []

        return self.thetha_list  # returning final weights


class nn_func():
    def __init__(self):
        w = np.random.rand(3, 2)
