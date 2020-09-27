'''
@ author : kexin tang
@ date   : 25/Sep/2020

'''
import numpy as np
import matplotlib.pyplot as plt

# activate function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# loss function
def loss(y, label):
    loss = 1.0/(2*num_observations) * np.sum(np.abs(y-label))
    return loss

# the forward and backward process
def perceptron(x, label, r = 0.0065, lr = 0.01):
    w = np.zeros(x.shape[1])    # add a column as bias
    i = 1
    while(1):
        y = sigmoid(np.dot(w, x.T))
        l = loss(y, label)
        if l>=r:
            w = w + lr * np.dot((label - y), x)     # the method for backward
        else:
            break
        if i%100 == 0:
            print("iter: " + str(i) + "\tloss: " + str(l))  # log for training
        i+=1
    return w


def show_clf(X, Y, w, num_observations=500):
    xx = np.linspace(-4, 5, num_observations)
    a = w[0]
    b = w[1]
    c = w[2]
    yy = -(a/b)*xx - (c/b)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.2)
    plt.plot(xx, yy, c='r')
    plt.show()


if __name__ == "__main__":
    np.random.seed(12)
    num_observations = 500

    a1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    a2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((a1, a2)).astype(np.float32)      # X[:,0] x-axis    X[:,1] y-axis
                                                    # shape: (2*num_observations, 2)
    X = np.column_stack((X, np.ones(2 * num_observations))) # add a column to represent bias
    Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))  # label
                                                                            # shape: (2*num_observations,)
    W = perceptron(X, Y)
    print(W)
    show_clf(X, Y, W, num_observations)     # plot the figure
