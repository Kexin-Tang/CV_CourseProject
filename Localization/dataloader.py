import numpy as np
import cv2
import os
from crop import Random_Crop

'''
    @ read img and gt from file whose path is "rote/className" and "rote/className_gt.txt"

    input:  rote        ->  absolute rote path
            className   ->  folder name(classes name)
    output: imgs, gts   ->  imgs and ground truth
'''
def readData(rote, className, times=5):
    random_crop = Random_Crop()

    imgs = []  # img's pixels
    gts = []

    imgs_crop = []  # img's pixels
    gts_crop = []

    img_file = os.path.join(rote, className)  # enter folder
    i = 0
    j = 0
    gt_file = os.path.join(rote, className + "_gt.txt")  # ground truth txt file
    gts = np.loadtxt(gt_file)  # get ground truth
    gts = gts[:, 1:]  # 1st column is index, which is redundant
    gts = gts.tolist()
    for file in os.listdir(img_file):
        img = cv2.imread(img_file + '/' + file)  # read imgs(only first 180 imgs are valid, will process later)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # torch is RGB, opencv is BGR, need transfer
        imgs.append(np.array(img / 255, dtype=np.float))
        for _ in range(times):
            img_crop, gt_crop = random_crop.crop(img, gts[j])
            img_crop = np.array(img_crop / 255, dtype=np.float)
            gt_crop = gt_crop.numpy().astype(int).tolist()
            imgs_crop.append(img_crop)
            gts_crop.append(gt_crop)
        j += 1

    gts = np.array(gts)
    gts_crop = np.array(gts_crop)

    return imgs, gts, imgs_crop, gts_crop


'''
    @ process several classes data, return ndarray type dataset

    input:  class_list      ->  class name
            rote            ->  absolute rote path
    output: Data/GT         ->  training/testing data/gt, and 1~150 is training, 151~180 is testing
'''
def process(class_list, rote, times=5):
    trainData = []
    trainGT = []
    trainLabel = []
    testData = []
    testGT = []
    testLabel = []

    i = 0

    for className in class_list:
        imgs, gts, imgs_crop, gts_crop = readData(rote, className)

        trainData.append(imgs_crop[:150*times])
        trainGT.append(gts_crop[:150*times])
        trainLabel.append(np.array([i] * 150*times))

        testData.append(imgs[150:180])
        testGT.append(gts[150:180])
        testLabel.append(np.array([i] * 30))
        i += 1

    trainData = np.array(trainData)
    trainGT = np.array(trainGT)
    trainLabel = np.array(trainLabel)
    testData = np.array(testData)
    testGT = np.array(testGT)
    testLabel = np.array(testLabel)

    trainData = trainData.reshape(-1, 128, 128, 3)
    trainGT = trainGT.reshape(-1, 4)
    trainLabel = trainLabel.reshape(-1)

    testData = testData.reshape(-1, 128, 128, 3)
    testGT = testGT.reshape(-1, 4)
    testLabel = testLabel.reshape(-1)

    # very important! transfer numpy shape to tensor shape [batch, H, W, C] -> [batch, C, H, W]
    trainData = np.transpose(trainData, (0, 3, 1, 2))
    testData = np.transpose(testData, (0, 3, 1, 2))

    return trainData, trainGT, trainLabel, testData, testGT, testLabel


if __name__ == "__main__":
    class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']  # define the 5 classes
    rote = "your path"  # define the rote path

    trainData, trainGT, trainLabel, testData, testGT, testLabel = process(class_list, rote)
    print(trainData.shape)  # (1500*classNum, 3, 128, 128)
    print(trainGT.shape)  # (1500*classNum, 4)
    print(trainLabel.shape)
    print(testData.shape)  # (30 *classNum, 3, 128, 128)
    print(testGT.shape)  # (30 *classNum, 4)
    print(testLabel.shape)
