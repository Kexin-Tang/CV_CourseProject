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
        img = cv2.imread(img_file + '/' + file)     # read imgs(only first 180 imgs are valid, will process later)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # torch is RGB, opencv is BGR, need transfer
        img = np.array(img/255, dtype=np.float)
        imgs.append(img)

    gt_file = os.path.join(rote, className+"_gt.txt")   # ground truth txt file
    gts = np.loadtxt(gt_file)                           # get ground truth
    gts = gts[:, 1:]                                    # 1st column is index, which is redundant
    gts = np.array(gts)
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
    trainLabel = []
    testData = []
    testGT = []
    testLabel = []

    i = 0

    for className in class_list:
        imgs, gts = readData(rote, className)
        # first 150 imgs for training, 150~180 for testing
        trainData.append(imgs[:150])
        trainGT.append(gts[:150])
        trainLabel.append(np.array([i] * 150))

        testData.append(imgs[150:180])
        testGT.append(gts[150:180])
        testLabel.append(np.array([i] * 30))
        i+=1

    trainData = np.array(trainData)
    trainGT = np.array(trainGT)
    trainLabel = np.array(trainLabel)
    testData = np.array(testData)
    testGT = np.array(testGT)
    testLabel = np.array(testLabel)

    trainData = trainData.reshape(-1, 128, 128, 3)  # [5, 150, 128, 128, 3] -> [750, 128, 128, 3]
    trainGT = trainGT.reshape(-1, 4)
    trainLabel = trainLabel.reshape(-1)

    testData = testData.reshape(-1, 128, 128, 3)    # [5, 30, 128, 128, 3] -> [150, 128, 128, 3]
    testGT = testGT.reshape(-1, 4)
    testLabel = testLabel.reshape(-1)

    # very important! transfer numpy shape to tensor shape [batch, H, W, C] -> [batch, C, H, W]
    trainData = np.transpose(trainData, (0, 3, 1, 2))
    testData = np.transpose(testData, (0, 3, 1, 2))

    return trainData, trainGT, trainLabel, testData, testGT, testLabel



if __name__ == "__main__":
    class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']     # define the 5 classes
    rote = "/home/kxtang/CV/tiny_vid" # define the rote path

    trainData, trainGT, trainLabel, testData, testGT, testLabel = process(class_list, rote)
    print(trainData.shape)  # (150*classNum, 3, 128, 128)
    print(trainGT.shape)    # (150*classNum, 4)
    print(trainLabel.shape) # (150*classNum, )
    print(testData.shape)   # (30 *classNum, 3, 128, 128)
    print(testGT.shape)     # (30 *classNum, 4)
    print(testLabel.shape)  # (30*classNum, )

