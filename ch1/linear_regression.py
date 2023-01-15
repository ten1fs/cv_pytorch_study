import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入相关数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 绘制数据
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo')
plt.show()

w = torch.tensor([-1.], requires_grad=True)
b = torch.tensor([0.], requires_grad=True)


# 构建模型
# y = w * x + b
def linear_model(x):
    return x * w + b


# 定义损失函数
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)


# 利用梯度下降法进行迭代训练
lr = 1e-2
for e in range(10):
    y_ = linear_model(x_train)
    # 计算损失
    loss = get_loss(y_, y_train)
    loss.backward()
    # 更新参数
    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data
    print('epoch: {}, loss: {}'.format(e, loss))
    # 梯度置零
    w.grad.zero_()
    b.grad.zero_()

# 绘制预测结果
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()
