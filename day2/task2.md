# 设立计算图并自动计算
## 1.numpy和pytorch实现梯度下降法
     a.设定初始值
     b.求取梯度
     c.在梯度方向上进行参数的更新

```
import numpy as np
import torch

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params, grads):
        params -= self.lr * grads


def check_optim(optim_numpy, optim_torch, p, p_torch):
    """
    check with y = p * x^2
    optim param p
    """
    x_size = 5
    x = np.random.random(x_size)
    x_torch = torch.tensor(x, requires_grad=True)

    dxi_numpy_list = []
    for i in range(x_size):
        yi_numpy = p * x[i] ** 2
        dxi_numpy = 2 * p * x[i]
        dxi_numpy_list.append(dxi_numpy)

        da = x[i] ** 2
        optim_numpy(p, da)

    for i in range(x_size):
        yi_torch = p_torch * x_torch[i] ** 2
        optim_torch.zero_grad()
        yi_torch.backward()
        optim_torch.step()

    print(np.array(dxi_numpy_list))
    print(x_torch.grad.data.numpy())

a_numpy = np.array(1.2)
a_torch = torch.tensor(a_numpy, requires_grad=True)
sgd_numpy = SGD(0.1)
sgd_torch = torch.optim.SGD([a_torch], lr=0.1)
check_optim(sgd_numpy, sgd_torch, a_numpy, a_torch)

```

## 2.numpy和pytorch实现线性回归

```
import numpy as np
import torch.utils.data
import torch as t
import matplotlib.pyplot as plt
t.manual_seed(1)    # reproducible
fig = plt.figure()
# 首先造出来一些带噪声的数据
N = 1100
x = t.linspace(0, 10, N)
y = 10*x+5+t.rand(N)*5
ax1 = fig.add_subplot(211)  # 高2 宽2 图号1
ax1.plot(x.numpy(), y.numpy())
# 从Tensor构建DataSet对象
data = t.utils.data.TensorDataset(x, y)
# 随机分割数据集，测试集
train, test = t.utils.data.random_split(data, [1000, 100])
# 建立dataloader shuffle来使得每个epoch前打乱数据
trainloader = t.utils.data.DataLoader(train, batch_size=100, shuffle=True)
testloader = t.utils.data.DataLoader(test, batch_size=100, shuffle=True)

# 定义模型参数 for y_head=w@x+b
w = t.rand(1, 1, requires_grad=True)
b = t.zeros(1, 1, requires_grad=True)

optimizer = t.optim.SGD([w, b], lr=0.03, momentum=0.6)
#optimizer = t.optim.Adam([w,b],lr=10)
loss_his = []
batch_loss = 0
for epoch in range(10):
    for i, (bx, by) in enumerate(trainloader):
        bx = bx.view(1, -1)  # torch.Size([100])->torch.Size([1,100])
        y_head = w@bx+b

        loss = 0.5*(y_head-by)**2
        loss = loss.mean()
        batch_loss += loss.item()

        optimizer.zero_grad()  # 先清除梯度
        loss.backward()
        optimizer.step()
        #print('training:epoch {} batch {}'.format(epoch,i))
    loss_his.append(batch_loss)
    batch_loss = 0


def main():
    print('w={},b={}'.format(w.item(), b.item()))
    y_head = w*x+b
    ax1.plot(x.numpy(), y_head.detach().numpy().flatten())
    ax1.set_title('fit')
    print('final_loss={}'.format(loss_his[-1]))
    ax2 = fig.add_subplot(212)  # 高2 宽2 图号2
    ax2.set_title('loss per epoch (log)')
    ax2.plot(np.log(loss_his))
    plt.show()


if __name__ == '__main__':
    main()

```

## 3.pytorch实现一个简单的神经网络

逻辑回归
```
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```
