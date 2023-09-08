import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
}
data_dir = './chest_xray'

train_sets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_sets, batch_size=32, shuffle=True)
train_size = len(train_sets)
train_classes = train_sets.classes

val_sets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=32, shuffle=False)
val_size = len(val_sets)

test_sets = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_sets, batch_size=16, shuffle=False)
test_size = len(test_sets)

# print('train_size:', train_size)
# print('val_size:', val_size)
# print('test_size:', test_size)


#  CNN网络结构搭建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, (11, 11), stride=4, padding=2)
        self.conv2 = nn.Conv2d(48, 128, (5, 5), padding=2)
        self.conv3 = nn.Conv2d(128, 192, (5, 5), padding=2)
        self.conv4 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(192, 128, (3, 3), padding=1)
        self.max_pooling = nn.MaxPool2d((3, 3), stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


My_net = Net()
My_net = My_net.to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(My_net.parameters(), lr=0.001, weight_decay=0.02)

#  画出损失函数随着epoch的变化图
def show_loss(epoch_list, train_loss_list, val_loss_list):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, train_loss_list, color='blue', linestyle="-", label="Train_Loss")
    plt.plot(epoch_list, val_loss_list, color='red', linestyle="-", label="Val_Loss")
    plt.legend()
    plt.show()


#  画出准确率随着epoch的变化图
def show_accurate(epoch_list, train_accurate_list, val_accurate_list):
    plt.xlabel('epochs')
    plt.ylabel('accurate')
    plt.plot(epoch_list, train_accurate_list, color='blue', linestyle="-", label="Train_accurate")
    plt.plot(epoch_list, val_accurate_list, color='red', linestyle="-", label="Val_accurate")
    plt.legend()
    plt.show()

#  训练模型
def train(Epoch):
    best_loss = 1000
    epoch_list = [i for i in range(Epoch)]
    train_loss_list = []
    val_loss_list = []
    train_accurate_list = []
    val_accurate_list = []

    print('************************************Start training************************************')
    for epoch in range(Epoch):
        My_net.train()
        running_loss = 0.0
        accurate_train = 0
        for i, data in enumerate(train_loader):
            inputs, label = data
            inputs = inputs.to(device)  # 传入GPU
            label = label.to(device)
            outputs = My_net(inputs)
            optimizer.zero_grad()
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accurate_train += (outputs.argmax(1) == label).sum()
        running_loss = running_loss / len(train_loader)
        My_net.eval()  # 评估模型
        val_loss = 0  # 验证集的loss
        accurate_val = 0  # 验证集正确个数
        with torch.no_grad():
            for i, data1 in enumerate(val_loader):
                inputs, labels = data1
                inputs1 = inputs.to(device)
                labels1 = labels.to(device)
                outputs1 = My_net(inputs1)
                loss1 = loss_function(outputs1, labels1)
                val_loss += loss1.item()
                accurate_val += (outputs1.argmax(1) == labels1).sum()
            val_loss = val_loss / len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(My_net.state_dict(), './model.pt')
        print('epoch:', epoch + 1, '|train_loss:%.3f' % running_loss, 'val_loss:%.3f' % val_loss,
              'train_accurate:{}%.'.format(accurate_train / len(train_sets) * 100),
              'val_accurate:{}%.'.format(accurate_val / len(val_sets) * 100))
        train_loss_list.append(running_loss)
        print(type(accurate_train), type(len(train_sets)))
        accurate_train=accurate_train.cpu()
        accurate_train=accurate_train.numpy()
        train_accurate_list.append(accurate_train / len(train_sets))
        print(type(accurate_train), type(len(train_sets)))
        val_loss_list.append(val_loss)
        accurate_val=accurate_val.cpu()
        accurate_val=accurate_val.numpy()
        val_accurate_list.append(accurate_val / len(val_sets))
    print('Done')
    #  可视化
    show_loss(epoch_list, train_loss_list, val_loss_list)
    show_accurate(epoch_list, train_accurate_list, val_accurate_list)


def test():
    test_net = Net()
    test_net = test_net.to(device)
    test_net.eval()
    test_net.load_state_dict(torch.load('./model.pt'))
    print("************************************Model loaded************************************")
    accurate_test = 0  # 正确个数
    with torch.no_grad():
        for i, data1 in enumerate(test_loader):
            inputs, labels = data1
            inputs1 = inputs.to(device)
            labels1 = labels.to(device)
            outputs1 = test_net(inputs1)
            accurate_test += (outputs1.argmax(1) == labels1).sum()
    print('test_accurate:{}%'.format(accurate_test / test_size * 100))