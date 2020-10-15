import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import ast
import os

def drawPlot(path, savepath):
    x = np.linspace(1, 20, 60)

    loss_class_list= []
    loss_regre_list = []
    acc_list = []

    with open(path, 'r') as f:
        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_class_list = loss.reshape(-1, 1)

        loss = f.readline()
        loss = np.array(ast.literal_eval(loss))
        loss_regre_list = loss.reshape(-1, 1)

        acc = f.readline()
        acc_list = np.array(ast.literal_eval(acc))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, loss_class_list, 'r', ls=":", label='classification loss')
    ax.plot(x, loss_regre_list, 'b', ls="--", label='regression loss')
    ax.set_ylabel('loss')
    plt.legend(loc='center right')
    ax1 = ax.twinx()
    ax1.plot(acc_list, 'g', ls="-.", label='accuracy')
    ax1.set_ylabel('accuracy')
    plt.legend(loc='best')
    fig.show()
    fig.savefig(savepath +'.png')

def drawRect(source, path, eval_txt, name, className):
    IoU = []            # iou
    box = []            # predict rectangle
    classList = []      # class type

    rote = 'C:/Users/Tang KeXin/PycharmProjects/localization/tiny_vid'
    store_path = os.path.join(rote, name)

    with open(eval_txt, 'r') as f:
        name = f.readline()
        name = np.array(ast.literal_eval(name))
        classList = name.reshape(-1)

        box_float = f.readline()
        box_float = np.array(ast.literal_eval(box_float))       # transfer str to numpy array
        box = box_float.reshape(-1, 4)                          # reshape
        box = box.astype(int)                                   # change float to int
        box[box > 127] = 127                                    # limit the overflow data

        iou = f.readline()
        IoU = np.array(ast.literal_eval(iou))

    j = 0
    for name in className:
        print("start processing " + name)
        folder = os.path.join(rote, name)
        for id in range(151, 181):
            img_path = folder+'/000'+str(id)+'.JPEG'
            img = cv2.imread(img_path)
            cv2.rectangle(img, (box[j][0], box[j][1]), (box[j][2], box[j][3]), (0,255,0), 4)
            cv2.imwrite(store_path+'/' + className[classList[j]] + '/' +str(j)+'.JPEG', img)
            j+=1
    return IoU

if __name__ == "__main__":
    className = ['bird', 'car', 'dog', 'lizard', 'turtle']

    # drawPlot('./results/VGG_CEL2_1_loss_acc.txt', 'VGG_CEL2_1')
    # iou = drawRect('./tiny_vid/', './output/', './results/VGG_CEL2_1_eval.txt', 'eval_1', className)
    # print('Average IoU of 1:\t' + str(sum(iou) / len(iou)))
    #
    # drawPlot('./results/VGG_CEL2_1e-1_loss_acc.txt', 'VGG_CEL2_1e-1')
    # iou = drawRect('./tiny_vid/', './output/', './results/VGG_CEL2_1e-1_eval.txt', 'eval_1e-1', className)
    # print('Average IoU of 1e-1:\t' + str(sum(iou) / len(iou)))
    #
    # drawPlot('./results/VGG_CEL2_2e-2_loss_acc.txt', 'VGG_CEL2_2e-2')
    # iou = drawRect('./tiny_vid/', './output/', './results/VGG_CEL2_2e-2_eval.txt', 'eval_2e-2', className)
    # print('Average IoU of 2e-2:\t' + str(sum(iou)/len(iou)))
