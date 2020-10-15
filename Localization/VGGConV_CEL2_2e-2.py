from torchvision import transforms, models
from dataloader import process, readData
from Net import VGG16Net, VGG16ConV
import torch
import torch.nn as N
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import Variable
import torch.optim as optim
import pdb
import numpy as np

'''
    @ override torch.utils.data.DataSet

    input:  Data/GT/Label   ->  img pixels/box coordinate/label
'''


class MyDataLoader(Dataset):
    def __init__(self, Data, GT, Label):
        self.data = torch.from_numpy(Data)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.gt = torch.from_numpy(GT)
        self.gt = torch.tensor(self.gt, dtype=torch.float32)
        self.label = torch.from_numpy(Label)
        self.len = Data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data[item], self.gt[item], self.label[item]

class ConVDataLoader(Dataset):
    def __init__(self, Data, GT, Label):
        self.data = torch.from_numpy(Data)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        GT = GT.reshape(-1,4,1,1)
        self.gt = torch.from_numpy(GT)
        self.gt = torch.tensor(self.gt, dtype=torch.float32)
        Label = Label.reshape(-1,1,1)
        self.label = torch.from_numpy(Label)
        self.len = Data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data[item], self.gt[item], self.label[item]


def train(args, model, device, train_loader, optim, epoch):
    loss_classify_list = []
    loss_regression_list = []
    model.train()
    for idx, (data, box, label) in enumerate(train_loader):
        data, box, label = data.to(device), box.to(device), label.to(device)
        optim.zero_grad()
        c, r = model(data)

        loss_classify = CELoss(c, label.long())  # cross entropy loss for classify path
        loss_regression = SmoothL1(r, box)

        loss = loss_classify + 0.1 * loss_regression
        loss.backward()

        optim.step()

        if idx % 5 == 0:
            print('epoch: ' + str(epoch) + '\ttrain iter: ' + str(idx * len(data)) +
                  '\tclassify loss: ' + str(loss_classify.item()) + '\tregression loss: ' + str(loss_regression.item()))
            loss_classify_list.append(loss_classify.item())
            loss_regression_list.append(0.1 * loss_regression.item())
    return loss_classify_list, loss_regression_list


def test(args, model, device, test_loader):
    model.eval()
    loss = 0
    acc = 0

    with torch.no_grad():
        for data, box, label in test_loader:
            data, box, label = data.to(device), box.to(device), label.to(device)
            c, r = model(data)

            loss += CELoss(c, label.long()).item()  # sum of batch loss
            pred = c.argmax(dim=1, keepdim=True)  # get index of the max prob
            acc += pred.eq(label.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    acc = 100.0 * acc / len(test_loader.dataset)
    print("Average loss: " + str(loss) + "\tAcc: " + str(acc) + "\n")
    return acc


def IoU(pred, gt):
    b1_x1, b1_y1, b1_x2, b1_y2 = pred
    b2_x1, b2_y1, b2_x2, b2_y2 = gt

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    w = np.maximum(0, inter_rect_x2 - inter_rect_x1)
    h = np.maximum(0, inter_rect_y2 - inter_rect_y1)

    inter_area = w * h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def imgEval(rote, model, device, eval_loader):
    model.eval()
    className = []
    iou = []
    box_out = []
    for data, box, label in eval_loader:
        data, box, label = data.to(device), box.to(device), label.to(device)
        c, r = model(data)

        c = c.argmax(dim=1, keepdim=True)

        pred = r.cpu().detach().numpy().reshape(-1)
        box = box.cpu().detach().numpy().reshape(-1)

        className.append(list(i) for i in c.cpu().detach().numpy().reshape(-1))

        iou.append(IoU(pred, box))

        for i in pred:
            box_out.append(list(i))

    file = open(rote, 'w')
    file.write(str(className))
    file.write('\n')
    file.write(str(box_out))
    file.write('\n')
    file.write(str(iou))
    file.close()


def saveTxt(rote, loss_classify, loss_regression, acc):
    file = open(rote, 'w')
    file.write(str(loss_classify))
    file.write('\n')
    file.write(str(loss_regression))
    file.write('\n')
    file.write(str(acc))
    file.close()


if __name__ == "__main__":
    class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']  # define the 5 classes
    rote = "/home/kxtang/CV/tiny_vid"  # define the rote path
    classify_loss_train = []
    regression_loss_train = []
    acc = []

    # set the parameters in console and their default parameters
    parser = argparse.ArgumentParser(description='PyTorch CNN Localization')
    parser.add_argument('--batch', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--test_batch', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    args = parser.parse_args()  # load parameters

    torch.manual_seed(20)  # generate random seeds for shuffle dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get data from folders and txt files
    trainData, trainGT, trainLabel, testData, testGT, testLabel = process(class_list, rote)
    print("------------Data Load Finished !!------------")

    # create my dataset by override torch.utils.data.DataSet
    train_dataset = ConVDataLoader(trainData, trainGT, trainLabel)
    test_dataset = ConVDataLoader(testData, testGT, testLabel)

    # dataloader to pytorch network
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch, shuffle=True)
    eval_loader = DataLoader(dataset=test_dataset, shuffle=False)

    model = VGG16ConV().to(device)

    CELoss = N.CrossEntropyLoss()  # define the loss function
    SmoothL1 = N.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam SGD optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  # Momentum optimizer

    for epoch in range(1, args.epochs + 1):
        train_loss_classify, train_loss_regression = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        classify_loss_train.append(train_loss_classify)
        regression_loss_train.append(train_loss_regression)
        acc.append(test_acc)

    saveTxt('./VGGConV_CEL2_2e-2_loss_acc.txt', classify_loss_train, regression_loss_train, acc)

    imgEval('./VGGConV_CEL2_2e-2_eval.txt', model, device, eval_loader)



