import torch
import torch.autograd as autograd
import torch.nn as nn

class MyFun(torch.autograd.Function):
  def forward(self, inp):
      return inp

  def backward(self, grad_out):
    grad_input = grad_out.clone()
    print('Custom backward called!')
    return grad_input

class MyMod(nn.Module):
  def __init__(self):
    super(MyMod, self).__init__()
    self.myfun = MyFun()

  def forward(self, x):
    return self.myfun(x)

class MyMod2(nn.Module):
  def __init__(self):
    super(MyMod2, self).__init__()
    self.mymod = MyMod()

  def forward(self, x):
    return self.mymod(x)

mod = MyMod2()

y = autograd.Variable(torch.randn(1), requires_grad=True)
z = mod(y)
z.backward()

print type((1,))