# PyTorch实现L1，L2正则化以及Dropout
```
import torch
import numpy as np


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.train_flg = True
        self.mask = None

    def __call__(self, x, manual_mask=None, train_flg=True):
        if train_flg:
            if manual_mask is None:
                self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            else:
                self.mask = manual_mask
            out = x * self.mask / (1.0 - self.dropout_ratio)
            return out
        else:
            return x

    def backward(self, d_loss):
        dx = d_loss * self.mask / (1.0 - self.dropout_ratio)
        return dx


np.set_printoptions(precision=6, suppress=True, linewidth=120)
np.random.seed(12)
torch.random.manual_seed(3)

x_numpy = np.random.random((3, 7))
x_tensor = torch.tensor(x_numpy, requires_grad=True)

drop_out_numpy = Dropout(dropout_ratio=0.45)
drop_out_tensor = torch.nn.Dropout(p=0.45)

print("\n----- 训练阶段 -----")
train_flag = True
drop_out_tensor.train()

out_tensor = drop_out_tensor(x_tensor)
mask = out_tensor > 0
mask = mask.data.numpy()
out_numpy = drop_out_numpy(x_numpy, mask, train_flg=train_flag)

print("train mask : \n", mask)
print("train x : \n", x_numpy)
print("numpy out : \n", out_numpy)
print("tensor out : \n", out_tensor.data.numpy())

print("\n----- 反向传播 -----")
d_loss_numpy = np.random.random((3, 7))
d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)

dx_numpy = drop_out_numpy.backward(d_loss_numpy)
out_tensor.backward(d_loss_tensor)
dx_tensor = x_tensor.grad
print("dx_numpy : \n", dx_numpy)
print("dx_tensor : \n", dx_tensor.data.numpy())

print("\n----- 测试阶段 -----")
train_flag = False
drop_out_tensor.eval()

out_tensor = drop_out_tensor(x_tensor)
mask = out_tensor > 0
mask = mask.data.numpy()
out_numpy = drop_out_numpy(x_numpy, mask, train_flg=train_flag)

print("test mask : \n", mask)
print("test x : \n", x_numpy)
print("numpy out : \n", out_numpy)
print("tensor out : \n", out_tensor.data.numpy())
```
**结果:**
```
----- 训练阶段 -----
train mask :
 [[1 0 1 0 1 0 0]
 [0 0 1 1 1 0 1]
 [1 1 0 0 0 1 0]]
train x :
 [[0.154163 0.74005  0.263315 0.533739 0.014575 0.918747 0.900715]
 [0.033421 0.956949 0.137209 0.283828 0.606083 0.944225 0.852736]
 [0.002259 0.521226 0.552038 0.485377 0.768134 0.160717 0.76456 ]]
numpy out :
 [[0.280296 0.       0.478755 0.       0.0265   0.       0.      ]
 [0.       0.       0.249471 0.516052 1.101969 0.       1.550428]
 [0.004108 0.947684 0.       0.       0.       0.292212 0.      ]]
tensor out :
 [[0.280296 0.       0.478755 0.       0.0265   0.       0.      ]
 [0.       0.       0.249471 0.516052 1.101969 0.       1.550428]
 [0.004108 0.947684 0.       0.       0.       0.292212 0.      ]]

----- 反向传播 -----
dx_numpy :
 [[0.037836 0.       0.211405 0.       1.220823 0.       0.      ]
 [0.       0.       1.277495 0.595581 0.60845  0.       1.135604]
 [1.727843 1.39541  0.       0.       0.       0.728421 0.      ]]
dx_tensor :
 [[0.037836 0.       0.211405 0.       1.220823 0.       0.      ]
 [0.       0.       1.277495 0.595581 0.60845  0.       1.135604]
 [1.727843 1.39541  0.       0.       0.       0.728421 0.      ]]

----- 测试阶段 -----
test mask :
 [[1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1]]
test x :
 [[0.154163 0.74005  0.263315 0.533739 0.014575 0.918747 0.900715]
 [0.033421 0.956949 0.137209 0.283828 0.606083 0.944225 0.852736]
 [0.002259 0.521226 0.552038 0.485377 0.768134 0.160717 0.76456 ]]
numpy out :
 [[0.154163 0.74005  0.263315 0.533739 0.014575 0.918747 0.900715]
 [0.033421 0.956949 0.137209 0.283828 0.606083 0.944225 0.852736]
 [0.002259 0.521226 0.552038 0.485377 0.768134 0.160717 0.76456 ]]
tensor out :
 [[0.154163 0.74005  0.263315 0.533739 0.014575 0.918747 0.900715]
 [0.033421 0.956949 0.137209 0.283828 0.606083 0.944225 0.852736]
 [0.002259 0.521226 0.552038 0.485377 0.768134 0.160717 0.76456 ]]
```
