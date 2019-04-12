# day4

```
import torch.nn.functional as F 
import torch.nn.init as init 
import torch
import numpy as np
import matplotlib.pyplot as plt

xy=np.loadtxt('diabetes.csv',dtype=np.float32,delimiter=',')

x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])

print(x_data.data.shape)
print(y_data.data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=torch.nn.Linear(8,6)
        self.l2=torch.nn.Linear(6,4)
        self.l3=torch.nn.Linear(4,1)
    def forward(self, x):
        x=F.relu(self.l1(x))
        x=F.dropout(x,p=0.5)
        x=F.relu(self.l2(x))
        x=F.dropout(x,p=0.5)
        x=F.sigmoid(self.l3(x))
        return x
net=Model()
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
loss_fuc=torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(2000):
    out=net(x_data)
    loss=loss_fuc(out,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y_data.data.numpy()
        plt.scatter(x_data.data.numpy()[:, 0], x_data.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```
