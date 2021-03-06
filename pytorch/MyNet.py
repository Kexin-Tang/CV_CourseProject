import torch
import numpy as np
import torch.nn as N
import torch.nn.functional as F
from torchvision import datasets, transforms    # mnist source and transfer strategy
import torch.optim as optim                     # optimal strategy
import argparse                                 # console parameters setting

class MyNet(N.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = N.Conv2d(1, 20, 5)     # Conv layer: (input, output, kernel)
        self.conv2 = N.Conv2d(20, 50, 5)
        self.fc1 = N.Linear(4*4*50, 500)    # Full Connected layer: (input, output)
        self.fc2 = N.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # activate function
        x = F.max_pool2d(x, 2, 2)   # max pooling, size (2,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)      # transfer matrix into 1-D list
        x = F.relu(self.fc1(x))     # full connected layer
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optim, epoch):
    loss_train = []
    model.train()
    # iter the train data, get the index, data and label
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device) # put data to device (cpu)
        optim.zero_grad()
        y = model(data)

        loss = F.cross_entropy(y, label)    # cross entropy loss
        loss.backward()                     # backward prop
        optim.step()

        if idx % 100 == 0:
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
            loss += F.cross_entropy(y, label, reduction='sum').item()   # sum of batch loss
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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()          # load parameters
    torch.manual_seed(5)                # generate random seeds for cpu to shuffle dataset
    device = torch.device("cpu")        # use cpu

    print("data load starting...")
    # load mnist from torchvision and normalize data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.test_batch, shuffle=True)
    print("data load finished!")

    model = MyNet().to(device)     # use cpu for training and testing

    # different optimizers for training
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)   # Momentum SGD optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)        # Adam SGD optimizer
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8)   # Adagrad SGD optimizer

    # train and test
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        loss_train.append(train_loss)
        acc.append(test_acc)

    # save the model parameters for fine-tune
    torch.save(model.state_dict(), "./models/MyNet_adagrad.pt")

    # save the loss and acc
    saveTxt('./results/MyNet_adagrad.txt', loss_train, acc)
