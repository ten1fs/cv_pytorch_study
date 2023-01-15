import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

# 进行运算
y = x + 2
print(y)
print(y.grad_fn)

# 叶子结点的grad_fn为None
print(x.is_leaf, y.is_leaf)  # x是叶子，y不是叶子

a = torch.randn(2, 2)  # 默认requires_grad=False
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

torch.manual_seed(10)  # 设置随机数
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
y.backward(retain_graph=True)
print(w.grad)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward()
print(x.grad)

x = torch.ones(1, requires_grad=True)
print(x.data)
print(x.data.requires_grad)
y = 2 * x
x.data *= 100  # 只是改变了值，不会记录在计算图中，所以不影响梯度传播
y.backward()
print(x)
print(x.grad)
