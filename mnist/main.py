'''
@ author :  kexin tang
@ date   :  30/Sep/2020
'''
import numpy as np
import matplotlib.pyplot as plt
from Net import Net
from mnist import load_mnist

# get the data and label
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# init the model
# 784 -> 256 -> 64 -> 16 -> 10
net = Net(input_size=784, hidden_size_1=256, hidden_size_2=64,
          hidden_size_3=16,  output_size=10, weight_init_std=0.01)

# training parameters
epoch = 50
batch_size = 100
lr = 0.5

train_size = x_train.shape[0]  # 60000
iter_per_epoch = max(train_size / batch_size, 1)  # 600

# loss and accuracy list
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter = int(iter_per_epoch * epoch)

# iter times
for i in range(iter):
    # random choose (batch_size,) data set
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = net.predict(x_batch)
    t_batch = t_train[batch_mask]
    grad = net.gradient(x_batch, t_batch)

    # update weights and bias
    for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4'):
        net.dict[key] -= lr * grad[key]
    # calculate the loss
    loss = net.loss(y_batch, t_batch)

    # store the accuracy and loss per iter_per_epoch
    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        print('iter: ' + str(i) + '\tloss: ' + str(loss) + '\ttest acc: ' + str(test_acc))


#   plot the figure about accuracy and loss
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.subplot(211)
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.subplot(212)
plt.plot(x, train_loss_list, label='train loss', linestyle='-.')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('res.png')
plt.show()
