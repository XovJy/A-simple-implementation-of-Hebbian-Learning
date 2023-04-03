import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

torch.set_printoptions(threshold=np.inf)


class HebbianLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(HebbianLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.weight, a=1)  # kaiming初始化
        # print(self.weight)

    def forward(self, x):
        output = F.linear(x, self.weight)

        hebb_term = torch.mm(output.t(), x)
        y_2 = torch.mul(output, output)
        y_2 = torch.mean(y_2, dim=0, keepdim=True)
        regu_term = torch.mul(self.weight.data, y_2.t())
        d_ws = 0.0001 * (hebb_term - regu_term)
        self.weight.data -= d_ws
        return output



class BackpropLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BackpropLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.fc(x)
        return output


class HebbianBackpropNet(nn.Module):
    def __init__(self):
        super(HebbianBackpropNet, self).__init__()
        self.layer1 = HebbianLayer(784, 2000)
        self.layer2 = BackpropLayer(2000, 10)
        # self.layer2 = HebbianLayer(2000, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)
        # return x


# 加载MNIST
batch_size = 64
train_dataset = datasets.MNIST(root='../datasets', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../datasets', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络
Net = HebbianBackpropNet()
optimizer = torch.optim.Adam(Net.layer2.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练网络
for epoch in range(100):
    train_acc = 0
    train_loss = 0
    outputs = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        optimizer.zero_grad()

        outputs = Net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.data
        predict = torch.max(outputs, 1)[1]
        num_correct = (predict == labels).sum().item()
        acc = num_correct / images.shape[0]
        train_acc += acc
    # print(outputs)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(epoch + 1, train_loss / len(train_loader),
                                                                        train_acc / len(train_loader)))



