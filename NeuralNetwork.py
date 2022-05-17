import numpy as np
import functions as F


class Model():
    def __init__(self, data, labels, minibatch=1, epochs=1, learning_rate=0.001):
        self.data = data
        self.labels = labels
        self.mini_batch = minibatch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_labels = None

        # TODO: initialize weights
        self.thetha_list = []
        # TODO: initialize weights

        self.grad_x_list = []
        self.x_list = []

    def forward(self, x):
        """
        :param x:
        :return:
        """
        self.x_list.append(x)
        curr_w = None
        for layer in self.thetha_list[:-1]:
            curr_w = layer[0]
            curr_b = layer[1]
            print("before relu", x.shape)
            x = F.relu(np.add(curr_w @ x, curr_b))
            print("after relu", x.shape)
            self.grad_x_list.append(np.array(curr_w.T @ F.grad_relu(x)))
            self.x_list.append(np.array(x))
        # get the weights of the last layer (softmax weights)
        softmax_theta = self.thetha_list[len(self.thetha_list) - 1]
        curr_w = softmax_theta[0]
        curr_b = softmax_theta[1]
        self.grad_x_list.append(
            F.gradient_softmax(x, (curr_w, curr_b), self.batch_labels, wrt='x'))
        x = F.softmax_func(x, curr_w, self.batch_labels)

        return x

    def add_hidden_layer(self, m, entries, res=False):
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

    def add_softmax_layer(self, entries, m, res=False):
        w = np.random.rand(m, entries)
        b = np.random.rand(1, entries)
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
        print(last_theta[0].shape)
        print("lallaa")
        print(len(self.thetha_list))
        print(len(self.x_list))
        print(len(self.grad_x_list))



        print("lalala")
        grad_w = F.gradient_softmax(last_x, last_theta,
                                       c=self.batch_labels,
                                       wrt='w')  # gradient of the loss function, last_theta(0) is w
        grad_b = F.gradient_softmax(last_x, last_theta,
                                       c=self.batch_labels,
                                       wrt='b')  # gradient of the loss function, last_theta(0) is w

        jacob_gradients.insert(0, [grad_w, grad_b])

        # update last layer
        v = self.grad_x_list[layer_num]
        print("first v", v.shape)
        for curr_theta in reversed(self.thetha_list[:len(self.thetha_list)-1]):  # TODO check indexes
            layer_num -= 1
            curr_w, curr_b = curr_theta
            curr_x = self.x_list[layer_num]
            # print("curr_x", curr_x)
            # print((F.grad_relu(curr_x)).shape)
            grad_w = np.multiply(F.grad_relu(np.add(curr_w @ curr_x, curr_b)),v) @ curr_x.T  # w.r.t W, J * v
            # print(grad_w)
            print("v shapeeeeeeeeee", v.shape)
            grad_b = np.sum(np.multiply(F.grad_relu(np.add(curr_w @ curr_x, curr_b)),v),axis=1)  # w.r.t bias TODO CHECK IF WE ITS COLUMNS SUM
            # print(len(grad_w))
            # print(len(grad_b.shape))

            jacob_gradients.insert(0, ([grad_w, grad_b]))  # in the correct order theta1,theta2,...,thetalosss
            print("shapes")
            print(self.grad_x_list[layer_num].shape)
            if layer_num == 0:
                break
            v = (self.grad_x_list[layer_num].T @ v) # add the next derivative to v for the backpropagation
            print("v newwww shape",v.shape)
        return jacob_gradients  # the gradients in rows -df1-,-df2- and so on

    def update_theta(self, jacob_gradients):
        for index, grad in enumerate(jacob_gradients):
            # print("shape of grad", grad.shape)
            # print(len(self.thetha_list[index]))
            self.thetha_list[index][0] = np.array(self.thetha_list[index][0]) - self.learning_rate * np.array(grad[0])
            self.thetha_list[index][1] = np.array(self.thetha_list[index][1]) - self.learning_rate * np.array(grad[1])

    # def resblock():
    #     pass

    def train(self):
        # loop over mini batches / epochs
        # TODO
        for epoch in range(self.epochs):
            indices_perm = np.random.permutation(self.data.shape[1])
            self.data = self.data[:, indices_perm]  # Shuffle the data
            print(self.labels)
            self.labels = np.array(self.labels)[indices_perm, :]
            # Iterations over the data (every epoch)
            batch_begin = 0
            batch_end = self.mini_batch - 1
            # until we still got enough images in the current epoch for the mini-batch
            while batch_end <= self.data.shape[1]:
                self.batch_labels = self.labels[batch_begin:batch_end + 1, :]
                print("self.batch_labels right now", self.batch_labels)
                self.forward(self.data[:, batch_begin:batch_end + 1])
                jacob_gradients = self.backward()
                self.update_theta(jacob_gradients)  # updating the weights

                # clear x list and grad list and update weights
                self.grad_x_list = []
                self.x_list = []

                batch_begin += self.mini_batch
                batch_end += self.mini_batch

        return self.thetha_list  # returning final weights


class nn_func():
    def __init__(self):
        w = np.random.rand(3, 2)
