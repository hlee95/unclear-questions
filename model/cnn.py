"""
This file implements the CNN for Part 1 of the project.
"""
import torch
from torch import nn
from torch.autograd import Variable

class CNN(nn.Module):
  def __init__(self, input_dim, output_dim, filter_width, activation=nn.Tanh(), use_cuda=False):
    super(CNN, self).__init__()
    # Create the different gates that we need.
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.filter_width = filter_width
    self.conv = nn.Conv1d(input_dim, output_dim, filter_width, padding=filter_width-1)
    self.activation = activation
    self.float_dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    if use_cuda:
        self.cuda()

  def forward(self, input, mask=None, return_average=True):
    """
    Compute using the equations given in the paper.
    """
    ct = self.conv(input)
    output = self.activation(ct[:,:,:-(self.filter_width-1)])

    if return_average:
        return torch.sum(output, 2)/torch.sum(mask).unsqueeze(0)
    else:
        # TODO: take into account how long the sentence is by using mask
        return output[:,:,-1]