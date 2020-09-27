'''
@ author: kexin tang
@ date  : 27/Sep/2020

'''
import numpy as np
import matplotlib.pyplot as plt
import random


'''
Generate the double moon figure
'''
def dbmoon(N=100, d=2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        # generate Rectangular data
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        # generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    db_moon = data[0:N, :]
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon


'''
sigmoid function
'''
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

'''
derivation of sigmoid
'''
def sigmoid_de(x):
    return x*(1.0-x)

'''
generate a m*n matrix
'''
def make_matrix(m,n,fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill]*n)
    return mat

'''
random function
'''
def rand(a,b):
    return (b-a)*random.random()+a


class BP_NET(object):
    def __init__(self):
        self.input_n = 0            # number of input neuron
        self.hidden_n = 0           # number of hidden neuron
        self.output_n = 0           # number of output neuron
        self.input_cells = []       # input data
        self.hidden_cells = []      # hidden data
        self.output_cells = []      # output data
        self.input_weights = []     # input-hidden weight
        self.output_weights = []    # hidden-output weight

    '''
    init parameters
    '''
    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        # random init weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        # random init weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

    '''
    generate the output
    '''
    def predict(self, inputs):
        # input layer
        for i in range(self.input_n-1):
            self.input_cells[i] = inputs[i]
        # hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    '''
    back propagation
    '''
    def back_propagate(self, data, label, lr):
        # calculate and get the output
        self.predict(data)
        output_deltas = [0.0] * self.output_n
        error = 0.0

        # loss in output
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_de(self.output_cells[o]) * error

        # output layer
        hidden_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.output_n):
                error += output_deltas[k] * self.output_weights[j][k]
            hidden_deltas[j] = sigmoid_de(self.hidden_cells[j]) * error
        # hidden layer
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += lr * change
        # input layer
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += lr * change
        # update error
        error = 0
        for o in range(len(label)):
            for k in range(self.output_n):
                error += 0.5 * (label[o] - self.output_cells[k]) ** 2

        return error

    '''
    train the net
    
    input:  datas   ->  input data
            labels  ->  ground truth
            r       ->  min loss
            lr      ->  learning rate
            max...  ->  max iter times
    '''
    def train(self, datas, labels, r, lr, max_iter=2000):
        i=0
        while(i<max_iter):
            i+=1
            error = 0.0
            for j in range(len(datas)):
                data = datas[j]
                label = labels[j]
                error += self.back_propagate(data, label, lr)
            if (i % 200 == 0):
                print("iter: " + str(i) + "\tloss: " + str(error))
            if error<=r:
                print("iter: " + str(i) + "\tloss: " + str(error))
                print("finish!")
                break

    '''
    test new input
    '''
    def test(self):

        N = 200
        d = -4
        r = 10
        width = 6

        # generate double moon figure
        data = dbmoon(N, d, r, width)

        input_cells = np.array([np.reshape(data[0:2 * N, 0], len(data)), np.reshape(data[0:2 * N, 1], len(data))]).transpose()

        labels_pre = [[1.0] for y in range(1, 201)]
        labels_pos = [[0.0] for y in range(1, 201)]

        labels = labels_pre + labels_pos

        self.setup(2, 5, 1)                                         # init the size of the net
        self.train(input_cells, labels, 0.1, 0.05, max_iter=2000)   # train the net

        test_x = []
        test_y = []
        test_p = []

        y_p_old = 0

        for x in np.arange(-15., 25., 0.1):
            for y in np.arange(-10., 10., 0.1):
                y_p = self.predict(np.array([x, y]))

                if (y_p_old < 0.5 and y_p[0] > 0.5):
                    test_x.append(x)
                    test_y.append(y)
                    test_p.append([y_p_old, y_p[0]])
                y_p_old = y_p[0]

        # draw the figure
        plt.plot(test_x, test_y, 'g--')
        plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2 * N, 0], data[N:2 * N, 1], 'b*')
        plt.show()


if __name__ == '__main__':
    nn = BP_NET()
    nn.test()

