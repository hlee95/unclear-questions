"""
This file implements the CNN for Part 1 of the project.
"""
import torch
from torch import nn
from torch.autograd import Variable

class CNNEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, filter_width, activation=nn.Tanh(),
               use_cuda=False, return_average=True):
    super(CNNEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.filter_width = filter_width
    self.return_average = return_average
    self.conv = nn.Conv1d(input_dim, output_dim, filter_width,
                          padding=filter_width-1)
    self.activation = activation
    self.float_dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    if use_cuda:
        self.cuda()

  def forward(self, input, mask=None):
    """
    Compute using the equations given in the paper.
    """
    ct = self.conv(input)
    # output is batch_size x output_dim x max_num_words
    # mask is batch_size x max_num_words
    output = self.activation(ct[:,:,:-(self.filter_width-1)])
    masked_output = output*mask[:, None, :]

    if self.return_average:
      ans = torch.sum(masked_output, 2)/(torch.sum(mask, 1)[:, None])
      return ans
    else:
      max_num_words = output.size()[2]
      batch_size = output.size()[0]

      last_h = Variable(torch.zeros(batch_size, self.output_dim)
        .type(self.float_dtype))
      for t in range(max_num_words):
        last_h = (1-mask[:,t])[:,None]*last_h.clone() + \
                 (mask[:,t])[:,None]*masked_output[:,:,t].clone()

      return last_h

  def run_all(self, input, mask=None):
    """
    This is just so that LSTMEncoder and CNNEncoder expose the same interface.
    """
    return self.forward(input, mask)

