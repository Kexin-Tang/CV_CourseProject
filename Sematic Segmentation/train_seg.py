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
random.seed(0)

from pspnet import PSPNet

'''
    @ get the img name list
'''
readvdnames = lambda x: open(x).read().rstrip().split('\n')


'''
    @ overload dataloader __init__, __getitem__ and __len__
'''
class TinySegData(Dataset):
    def __init__(self, db_root="TinySeg", img_size=256, phase='train'):
        classes = ['person', 'bird', 'car', 'cat', 'plane', ]
        seg_ids = [1, 2, 3, 4, 5]

        templ_image = db_root + "/JPEGImages/{}.jpg"
        templ_mask = db_root + "/Annotations/{}.png"

        ids = readvdnames(db_root + "/ImageSets/" + phase + ".txt")

        # build training and testing dbs
        samples = []
        for i in ids:
            samples.append([templ_image.format(i), templ_mask.format(i)])
        self.samples = samples
        self.phase = phase
        self.db_root = db_root
        self.img_size = img_size

        # data agumentation in brightness, contrast and etc.
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)

        if not self.phase == 'train':
            print ("resize and augmentation will not be applied...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)

    def get_train_item(self, idx):
        sample = self.samples[idx]                  # sample = [img, mask]
        image = Image.open(sample[0])

        if random.randint(0, 1) > 0:
            image = self.color_transform(image)
        image = np.asarray(image)[..., ::-1]        # to BGR
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)  # mask of img

        image = image.astype(np.float32)
        image = image / 127.5 - 1                   # normalize to -1~1

        if random.randint(0, 1) > 0:
            image = image[:, ::-1, :]               # HWC
            seg_gt = seg_gt[:, ::-1]

        # random crop to 256x256
        height, width = image.shape[0], image.shape[1]
        if height == width:
            miny, maxy = 0, 256
            minx, maxx = 0, 256
        elif height > width:
            miny = np.random.randint(0, height-256)
            maxy = miny+256
            minx = 0
            maxx = 256
        else:
            miny = 0
            maxy = 256
            minx = np.random.randint(0, width-256)
            maxx = minx+256
        image = image[miny:maxy, minx:maxx, :].copy()
        seg_gt = seg_gt[miny:maxy, minx:maxx].copy()

        if self.img_size != 256:
            new_size = (self.img_size, self.img_size)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            seg_gt = cv2.resize(seg_gt, new_size, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))      # To CHW which suitable to torch

        return image, seg_gt, sample

    def get_test_item(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample[0])
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

        image = image.astype(np.float32)
        image = image / 127.5 - 1        # -1~1
        image = np.transpose(image, (2, 0, 1))

        return image, seg_gt, sample


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
    IMG_SIZE = 128
    print ("=> the training size is {}".format(IMG_SIZE))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='train'), batch_size=32, shuffle=True, num_workers=8)
    #val_loader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)

    model = PSPNet(n_classes=6, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()


    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    # model.load_state_dict(torch.load("ckpt_seg/epoch_79_iou0.88.pth"))

    ckpt_dir = "ckpt_seg"
    mkdirs(ckpt_dir)
    epoch = 80

    for i in range(0, epoch):
        # train
        model.train()
        epoch_iou = []
        epoch_start = time.time()
        for j, (images, seg_gts, rets) in enumerate(train_loader):
            images = images.to(device)
            seg_gts = seg_gts.to(device)
            optimizer.zero_grad()

            seg_logit = model(images)
            loss_seg = criterion(seg_logit, seg_gts.long())
            loss = loss_seg
            loss.backward()
            optimizer.step()

            # epoch_loss += loss.item()
            if j % 10 == 0:
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                confusion_matrix = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=6)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IU = IU_array.mean()

                log_str = "[E{}/{} - {}] ".format(i, epoch, j)
                log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, ".format(loss_seg.item(), mean_IU)
                print (log_str)

                images_np = np.transpose((images.cpu().numpy()+1)*127.5, (0, 2, 3, 1))
                n, h, w, c = images_np.shape
                images_np = images_np.reshape(n*h, w, -1)[:, :, 0]
                seg_preds_np = seg_preds_np.reshape(n*h, w)
                visual_np = np.concatenate([images_np, seg_preds_np*40], axis=1)       # NH * W
                cv2.imwrite('visual.png', visual_np)
                epoch_iou.append(mean_IU)

        epoch_iou = np.mean(epoch_iou)
        epoch_end = time.time()
        epoch_time = round(epoch_end-epoch_start, 2)
        print ("=> This epoch costs {}s...".format(epoch_time))
        if i % 10 == 0 or i ==  epoch-1:
            print ("=> saving to {}".format("{}/epoch_{}_iou{:0.2f}.pth".format(ckpt_dir, i, epoch_iou)))
            torch.save(model.state_dict(), "{}/epoch_{}_iou{:0.2f}.pth".format(ckpt_dir, i, epoch_iou))

