# day1
## 1.Pytorch的基本概念：
    PyTorch是使用GPU和CPU优化的深度学习张量库。早先的版本基于torch，当前的版本基于caffe，由facebook支持开发。
    为什么选择pytroch？支持GPU；动态神经网络；Python 优先；命令式体验；轻松扩展。

## 2.开发环境 
   python3.7+anaconda5.3，cuda10.0，pycharm community百度搜索根据自己电脑配置下载相应的版本安装，cuda安装出现报错，参考https://blog.csdn.net/bingo_6/article/details/80114440和https://blog.csdn.net/zzpong/article/details/80282814 
## 3.Pytorch安装
   在官网https://pytorch.org/上根据自己的系统、环境和cuda版本 按照指示安装
## 4.代码示例
```
import torch
from torch.autograd import Variable

# train data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

criterion = torch.nn.MSELoss(size_average=False) # Defined loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Defined optimizer

# Training: forward, loss, backward, step
# Training loop
for epoch in range(50):
    # Forward pass
    y_pred = model(x_data)

    # Compute loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients
    optimizer.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])
```
