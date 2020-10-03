import numpy as np
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error

class Net:
    # random init weights and bias
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, weight_init_std):
        self.dict = {}
#         self.dict['w1'] = weight_init_std * np.random.randn(input_size, hidden_size_1)
#         self.dict['b1'] = np.zeros(hidden_size_1)
#         self.dict['w2'] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)
#         self.dict['b2'] = np.zeros(hidden_size_2)
#         self.dict['w3'] = weight_init_std * np.random.randn(hidden_size_2, hidden_size_3)
#         self.dict['b3'] = np.zeros(hidden_size_3)
#         self.dict['w4'] = weight_init_std * np.random.randn(hidden_size_3, output_size)
#         self.dict['b4'] = np.zeros(output_size)
        self.dict['w1'] = np.random.randn(input_size, hidden_size_1) / np.sqrt(input_size/2)
        self.dict['b1'] = np.zeros(hidden_size_1)
        self.dict['w2'] = np.random.randn(hidden_size_1, hidden_size_2) / np.sqrt(hidden_size_1/2)
        self.dict['b2'] = np.zeros(hidden_size_2)
        self.dict['w3'] = np.random.randn(hidden_size_2, hidden_size_3) / np.sqrt(hidden_size_2/2)
        self.dict['b3'] = np.zeros(hidden_size_3)
        self.dict['w4'] = np.random.randn(hidden_size_3, output_size) / np.sqrt(hidden_size_3/2)
        self.dict['b4'] = np.zeros(output_size)

    # predict
    def predict(self, x):
        w1, w2, w3, w4 = self.dict['w1'], self.dict['w2'], self.dict['w3'], self.dict['w4']
        b1, b2, b3, b4 = self.dict['b1'], self.dict['b2'], self.dict['b3'], self.dict['b4']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, w4) + b4
        y = softmax(a4)
        return y

    # calculate the loss
    def loss(self, y, t):
        t = t.argmax(axis=1)
        num = y.shape[0]
        s = y[np.arange(num), t]
        return -np.sum(np.log(s)) / num


    def gradient(self, x, t):
        w1, w2, w3, w4 = self.dict['w1'], self.dict['w2'], self.dict['w3'], self.dict['w4']
        b1, b2, b3, b4 = self.dict['b1'], self.dict['b2'], self.dict['b3'], self.dict['b4']
        grads = {}

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, w4) + b4
        y = softmax(a4)

        num = x.shape[0]
        dy = (y-t)/num
        grads['w4'] = np.dot(z3.T, dy)
        grads['b4'] = np.sum(dy, axis=0)

        da3 = np.dot(dy, w4.T)
        dz3 = sigmoid_grad(a3) * da3
        grads['w3'] = np.dot(z2.T, dz3)
        grads['b3'] = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, w3.T)
        dz2 = sigmoid_grad(a2) * da2
        grads['w2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, w2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        p = np.argmax(y, axis=1)
        q = np.argmax(t, axis=1)
        acc = np.sum(p == q) / len(y)
        return acc
