"""
This file implements the domain classifier for Part 2 of the project.

The gradient reversal layer is simply the identity in the forward
direction, and inverses the gradient and multiplies by a scaling factor
Lambda in the backwards direction.
"""
import torch
from torch import nn

class GradientReversalLayer(nn.Module):
  def __init__(self, Lambda, use_cuda=False):
    super(GradientReversalLayer, self).__init__()
    self.Lambda = Lambda

    if use_cuda:
      self.cuda()

  def forward(self, input):
    return input

  def backward(self, grad_output):
    return -self.Lambda * grad_output

