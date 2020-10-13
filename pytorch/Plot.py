import matplotlib.pyplot as plt
import numpy as np
import ast

def drawMyNet():
    x = np.linspace(1,10,60)
    x_acc = np.linspace(1, 10, 10)

    loss_adagrad = []
    acc_adagrad = []

    loss_adam = []
    acc_adam = []

    loss_momentum = []
    acc_momentum = []

    with open('./results/MyNet_adagrad.txt','r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_adagrad = loss.reshape(-1, 1)

        acc = f.readline()
        acc_adagrad = np.array(ast.literal_eval(acc))

    with open('./results/MyNet_adam.txt', 'r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_adam = loss.reshape(-1, 1)

        acc = f.readline()
        acc_adam = np.array(ast.literal_eval(acc))

    with open('./results/MyNet_momentum.txt', 'r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_momentum = loss.reshape(-1, 1)

        acc = f.readline()
        acc_momentum = np.array(ast.literal_eval(acc))

    plt.figure()
    plt.plot(x, loss_adagrad, color='blue', label='adagrad', ls='-.')
    plt.plot(x, loss_adam, color='red', label='adam', ls=':')
    plt.plot(x, loss_momentum, color='green', label='momentum', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('MyNet loss-epoch')
    plt.legend()
    plt.savefig('MyNet_loss.png')

    plt.figure()
    plt.plot(x_acc, acc_adagrad, color='blue', label='adagrad', ls='-.')
    plt.plot(x_acc, acc_adam, color='red', label='adam', ls=':')
    plt.plot(x_acc, acc_momentum, color='green', label='momentum', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('MyNet acc-epoch')
    plt.legend()
    plt.savefig('MyNet_acc.png')


def drawNet():
    x = np.linspace(1, 20, 100)

    x_acc = np.linspace(1, 20, 20)

    loss_vgg = []
    acc_vgg = []

    loss_res = []
    acc_res = []

    loss_res_adam = []
    acc_res_adam = []

    with open('./results/VGG16.txt','r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_vgg = loss.reshape(-1, 1)

        acc = f.readline()
        acc_vgg = np.array(ast.literal_eval(acc))

    with open('./results/Res50.txt', 'r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_res = loss.reshape(-1, 1)

        acc = f.readline()
        acc_res = np.array(ast.literal_eval(acc))


    with open('./results/Res50_Adam.txt', 'r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_res_adam = loss.reshape(-1, 1)

        acc = f.readline()
        acc_res_adam = np.array(ast.literal_eval(acc))

    plt.figure()
    plt.plot(x, loss_vgg, color='blue', label='vgg16(momentum)', ls='--')
    plt.plot(x, loss_vgg_adam, color='green', label='vgg16(adam)', ls=':')
    plt.plot(x, loss_res, color='red', label='res50(momentum)', ls='-')
    plt.plot(x, loss_res_adam, color='purple', label='res50(adam)', ls='-.')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('VGG16 Vs Res50')
    plt.legend()
    plt.savefig('vgg16_res50_loss.png')

    plt.figure()
    plt.plot(x_acc, acc_vgg, color='blue', label='vgg16', ls='--')
    plt.plot(x_acc, acc_vgg_adam, color='green', label='vgg16(adam)', ls=':')
    plt.plot(x_acc, acc_res, color='red', label='res50(momentum)', ls='-')
    plt.plot(x_acc, acc_res_adam, color='purple', label='res50(adam)', ls='-.')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('VGG16 Vs Res50')
    plt.legend()
    plt.savefig('vgg16_res50_acc.png')



if __name__ == "__main__":
    drawMyNet()
    drawNet()
