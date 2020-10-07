import torch
import numpy as np
import torch.nn as N
import torch.nn.functional as F
from torchvision import datasets, transforms, models    # mnist source and transfer strategy
import torch.optim as optim                             # optimal strategy
import argparse                                         # console parameters setting

class VGG16Net(N.Module):
    def __init__(self):
        super(VGG16Net,self).__init__()
        net = models.vgg16(pretrained=True)
        self.features = net
        self.classifier = N.Sequential(
            N.Linear(1000, 128),
            N.ReLU(),
            N.Dropout(),
            N.Linear(128, 10))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


def train(args, model, device, train_loader, optim, epoch):
    loss_train = []
    model.train()
    # iter the train data, get the index, data and label
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device) # put data to device (cpu)
        optim.zero_grad()
        y = model(data)

        loss = crit(y, label)    # cross entropy loss
        loss.backward()                        # backward prop
        optim.step()

        if idx % 50 == 0:
            print('epoch: ' + str(epoch) + '\ttrain iter: ' + str(idx*len(data)) + '\tloss: ' + str(loss.item()))
            loss_train.append(loss.item())
    return loss_train


def test(args, model, device, test_loader):
    model.eval()
    loss = 0
    acc = 0

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            y = model(data)
            loss += crit(y, label).item()   # sum of batch loss
            pred = y.argmax(dim=1, keepdim=True)                        # get index of the max prob
            acc += pred.eq(label.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    acc = 100.0*acc/len(test_loader.dataset)
    print("Average loss: " + str(loss) + "\tAcc: " + str(acc) + "\n")
    return acc



def saveTxt(rote, loss, acc):
    file = open(rote, 'w')
    file.write(str(loss))
    file.write('\n')
    file.write(str(acc))
    file.close()


if __name__ == "__main__":

    loss_train = []
    acc = []

    # set the parameters in console and their default parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()          # load parameters
    torch.manual_seed(5)                # generate random seeds for cpu to shuffle dataset
    device = torch.device("cpu")        # use cpu

    print("data load starting...")
    # load mnist from torchvision and normalize data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])),
        batch_size=args.batch, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])),
        batch_size=args.test_batch, shuffle=True)
    print("data load finished!")

    model = VGG16Net().to(device)
    crit = N.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  # Adam SGD optimizer

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        loss_train.append(train_loss)
        acc.append(test_acc)

    # save the model parameters for fine-tune
    torch.save(model.state_dict(), "./models/VGG16.pt")

    # save the loss and acc
    saveTxt('./results/VGG16.txt', loss_train, acc)