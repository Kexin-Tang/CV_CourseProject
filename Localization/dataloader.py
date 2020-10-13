import numpy as np
import cv2
import os

'''
    @ read img and gt from file whose path is "rote/className" and "rote/className_gt.txt"
    
    input:  rote        ->  absolute rote path
            className   ->  folder name(classes name)
    output: imgs, gts   ->  imgs and ground truth
'''
def readData(rote, className):
    imgs = []                                       # img's pixels
    img_file = os.path.join(rote, className)        # enter folder
    for file in os.listdir(img_file):
        img = cv2.imread(img_file + '\\' + file)    # read imgs(only first 180 imgs are valid, will process later)
        imgs.append(img)

    gt_file = os.path.join(rote, className+"_gt.txt")   # ground truth txt file
    gts = np.loadtxt(gt_file)                           # get ground truth
    gts = gts[:, 1:]                                    # 1st column is index, which is redundant
    return imgs, gts

'''
    @ process several classes data, return ndarray type dataset
    
    input:  class_list      ->  class name
            rote            ->  absolute rote path
    output: Data/GT         ->  training/testing data/gt, and 1~150 is training, 151~180 is testing
'''
def process(class_list, rote):
    trainData = []
    trainGT = []
    testData = []
    testGT = []

    for className in class_list:
        print('Loading ' + className + ' ...')
        imgs, gts = readData(rote, className)
        # first 150 imgs for training, 150~180 for testing
        trainData.append(imgs[:150])
        trainGT.append(gts[:150])
        testData.append(imgs[150:180])
        testGT.append(gts[150:180])
        print('Loading ' + className + 'finished!')
    trainData = np.array(trainData)
    trainGT = np.array(trainGT)
    testData = np.array(testData)
    testGT = np.array(testGT)
    trainData = trainData.reshape(-1, 128, 128, 3)
    trainGT = trainGT.reshape(-1, 4)
    testData = testData.reshape(-1, 128, 128, 3)
    testGT = testGT.reshape(-1, 4)
    return trainData, trainGT, testData, testGT



if __name__ == "__main__":
    ''' please remeber to change these params to your own dataset and absolute path '''
    class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']     # define the 5 classes
    rote = "XXX" # define the rote path

    trainData, trainGT, testData, testGT = process(class_list, rote)
    print(trainData.shape)  # (150*classNum, 128, 128, 3)
    print(trainGT.shape)    # (150*classNum, 4)
    print(testData.shape)   # (30 *classNum, 128, 128, 3)
    print(testGT.shape)     # (30 *classNum, 4)
