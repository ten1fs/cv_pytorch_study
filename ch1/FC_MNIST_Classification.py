import random
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import matplotlib.pyplot as plt


# 定义网络结构
class Net(nn.Module):
    def __init__(self, in_c=784, out_c=10):
        super(Net, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(in_c, 512)
        # 定义激活层
        self.act1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, out_c)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net()

# 准备数据集
# 训练集
train_set = mnist.MNIST('../data', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_set = mnist.MNIST('../data', train=False, transform=transforms.ToTensor(), download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=True)

# 可视化数据
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    idx = random.randint(0, len(train_set))
    digit_0 = train_set[idx][0].numpy()
    digit_0_image = digit_0.reshape(28, 28)
    ax.imshow(digit_0_image, interpolation='nearest')
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')
plt.show()

# 定义损失函数——交叉熵
criterion = nn.CrossEntropyLoss()
# 定义优化器——随机梯度下降
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)

# 开始训练
loss_list = []  # 记录训练损失
acc_list = []  # 记录训练精度
eval_loss_list = []  # 记录测试损失
eval_acc_list = []  # 记录测试精度
nums_epoch = 20  # 迭代次数
for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for batch, (img, label) in enumerate(train_data):
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
        # 前向传播
        out = net(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        if (batch + 1) % 200 == 0:
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                         batch + 1,
                                                                                         loss.item(),
                                                                                         acc))
        train_acc += acc
    loss_list.append(train_loss / len(train_data))
    acc_list.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img, label in test_data:
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        out = net(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_loss_list.append(eval_loss / len(test_data))
    eval_acc_list.append(eval_acc / len(test_data))
    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)
    ))

# 结果可视化
plt.figure()
plt.suptitle('Test', fontsize=12)

ax1 = plt.subplot(1, 2, 1)
ax1.plot(eval_loss_list)
ax1.set_title('Loss', fontsize=10, color='r')

ax2 = plt.subplot(1, 2, 2)
ax2.plot(eval_acc_list)
ax2.set_title('Acc', fontsize=10, color='b')

plt.show()
