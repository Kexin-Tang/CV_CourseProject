import random
import torch
import cv2
import numpy as np
import os


class Random_Crop(object):

    def crop(self, im, boxes_origin, sample_idx=6):
        top, bottom, left, right = (sample_idx,sample_idx,sample_idx,sample_idx)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)) # firstly, upsample picture
        boxes = [i+sample_idx for i in boxes_origin]

        self.random_getim = im, boxes
        imh, imw, _ = im.shape
        self.small_threshold = 1 / max(imw, imh)    # limit the minimum size
        w = 128   # the target size

        while True:
            h = w
            # random choose a square area
            x = random.randint(0, imw - w)
            y = random.randint(0, imh - h)
            roi = torch.Tensor([x, y, x + w, y + h])
            boxes = torch.Tensor(boxes)
            center = (boxes[:2] + boxes[2:]) / 2  # center point in origin picture

            mask = (center > roi[:2]) & (center < roi[2:])  # center point in new picture
            mask = mask[0] & mask[1]

            if not mask.any():  # 如果全为零，舍弃这个crop patch
                im, boxes  = self.random_getim
                imh, imw, _ = im.shape
                continue
                
            # Warning: first column is y, second is x
            img = im[y:y+h, x:x+w]
            boxes[0].add_(-x).clamp_(min=0, max=w)  # clamp 夹并在x,y之间
            boxes[1].add_(-y).clamp_(min=0, max=h)
            boxes[2].add_(-x).clamp_(min=0, max=w)
            boxes[3].add_(-y).clamp_(min=0, max=h)
            
            return img, boxes
