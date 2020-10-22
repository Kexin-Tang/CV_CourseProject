import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import os, time
import random
from train_seg import TinySegData
import pdb
from pspnet import PSPNet


"""
    @ Calcute the confusion matrix by given label and pred
    
    input:  gt_label    ->  the ground truth label
            pred_label  ->  the pred label
            class_num   ->  the number of class
    output: the confusion matrix
"""
def get_confusion_matrix(gt_label, pred_label, class_num):

        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        return confusion_matrix


def get_confusion_matrix_for_3d(gt_label, pred_label, class_num):
    confusion_matrix = np.zeros((class_num, class_num))

    for sub_gt_label, sub_pred_label in zip(gt_label, pred_label):
        sub_gt_label = sub_gt_label[sub_gt_label != 255]
        sub_pred_label = sub_pred_label[sub_pred_label != 255]
        cm = get_confusion_matrix(sub_gt_label, sub_pred_label, class_num)
        confusion_matrix += cm
    return confusion_matrix



if __name__ == "__main__":
    IoU = []
    
    print("Eval Process Starting...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_loader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)
    print("data loading finished...")

    model = PSPNet(n_classes=6).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    model.load_state_dict(torch.load("./ckpt_seg/epoch_79_iou0.85.pth"))   # the model storing path

    # eval
    model.eval()
    for j, (images, seg_gts, rets) in enumerate(val_loader):
        if j%100 == 0:
            print('{} sets finished...'.format(j))

        # load data to device
        images = images.to(device)
        seg_gts = seg_gts.to(device)

        # get prediction
        seg_logit = model(images)
        loss_seg = criterion(seg_logit, seg_gts.long())
        seg_preds = torch.argmax(seg_logit, dim=1)
        seg_preds_np = seg_preds.detach().cpu().numpy()
        seg_gts_np = seg_gts.cpu().numpy()

        confusion_matrix = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=6)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        x = [i for i in IU_array if i>0]
        IoU.append(np.mean(x))
    
    print('mIoU is {}'.format(np.mean(IoU)))
    with open('iou.txt', 'w') as f:
        f.write(str(IoU))
