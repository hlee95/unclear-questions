"""
This file implements the domain classifier for Part 2 of the project.

The gradient reversal layer is simply the identity in the forward
direction, and inverses the gradient and multiplies by a scaling factor
Lambda in the backwards direction.

This was useful: https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
"""
import torch
from torch import nn

class GradientReversalFunction(torch.autograd.Function):

  def __init__(self, Lambda):
    self.Lambda = Lambda

  def forward(self, input):
    return input.view_as(input)

  def backward(self, grad_output):
    # Multiply gradient by -self.Lambda
    return self.Lambda * grad_output.neg()

class GradientReversalLayer(nn.Module):
  def __init__(self, Lambda, use_cuda=False):
    super(GradientReversalLayer, self).__init__()

    self.Lambda = Lambda
    self.gradient_reversal_func = GradientReversalFunction(self.Lambda)

    if use_cuda:
      self.cuda()

  def forward(self, input):
    return self.gradient_reversal_func(input)

