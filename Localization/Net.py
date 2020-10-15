import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
import torchvision

'''
  full connected type
'''
class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net,self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential() # clean the vgg16 pretrained model's fc layer
        self.features = net

        # set the classification path for class classify
        self.classify_fc1 = nn.Linear(512*7*7, 256)
        self.classify_fc2 = nn.Linear(256, 32)
        self.classify_fc3 = nn.Linear(32, 5)
        # set the regression path for boxes prediction
        self.regression_fc1 = nn.Linear(512*7*7, 256)
        self.regression_fc2 = nn.Linear(256, 32)
        self.regression_fc3 = nn.Linear(32, 4)


    def forward(self, x):
        # warning:  torch shape is [batch, channel, height, weight]
        #           but x shape is [batch, height, weight, channel]
        #           so we need to change the shape of input x
        x = self.features(x)
        x = x.view(x.size(0),-1)

        # classification path
        c = F.relu(self.classify_fc1(x))
        c = F.relu(self.classify_fc2(c))
        c = self.classify_fc3(c)
        # regression path
        r = F.relu(self.regression_fc1(x))
        r = F.relu(self.regression_fc2(r))
        r = self.regression_fc3(r)

        return c,r

'''
  conv type
'''
class VGG16ConV(nn.Module):
    def __init__(self):
        super(VGG16ConV,self).__init__()
        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Load pretrained layers
        self.load_pretrained_layers()

        # classify path
        self.classify_conv1 = nn.Conv2d(512, 256, 7)
        self.classify_conv2 = nn.Conv2d(256, 64, 1)
        self.classify_conv3 = nn.Conv2d(64, 5, 1)

        # regression path
        self.regression_conv1 = nn.Conv2d(512, 128, 7)
        self.regression_conv2 = nn.Conv2d(128, 64, 1)
        self.regression_conv3 = nn.Conv2d(64, 32, 1)
        self.regression_conv4 = nn.Conv2d(32, 4, 1)



    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)
        
        # classify path
        c = F.relu(self.classify_conv1(out))
        c = F.relu(self.classify_conv2(c))
        c = self.classify_conv3(c)

        # regression path
        r = F.relu(self.regression_conv1(out))
        r = F.relu(self.regression_conv2(r))
        r = F.relu(self.regression_conv3(r))
        r = self.regression_conv4(r)

        return c,r

    '''
      set the pretrained params in former layers
    '''
    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

