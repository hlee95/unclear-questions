"""
This file implements the LSTM for Part 1 of the project.
"""
import torch
from torch import nn
from torch.autograd import Variable

class RNNLayer(nn.Module):
  def __init__(self, input_dim, hidden_dim, activation):
    super(RNNLayer, self).__init__()
    self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.h_activation = activation

  def forward(self, input, prev_hidden):
    """
    Run one step of the RNN.
    Returns the new hidden state.
    """
    hidden = self.i2h(torch.cat((input, prev_hidden), 0))
    hidden = self.h_activation(hidden)
    return hidden

class LSTM(nn.Module):
  def __init__(self, input_dim, output_dim, activation=nn.Tanh()):
    super(LSTM, self).__init__()
    # Create the different gates that we need.
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.i_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.f_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.o_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.z = RNNLayer(input_dim, output_dim, nn.Tanh())
    self.activation = activation

  def forward(self, input, prev_h, prev_c):
    """
    Compute using the equations given in the paper.
    Returns the new h and c vectors.
    """
    i_t = self.i_gate(input, prev_h)
    f_t = self.f_gate(input, prev_h)
    o_t = self.o_gate(input, prev_h)
    z_t = self.z(input, prev_h)
    c_t = i_t * z_t + f_t * prev_c
    h_t = o_t * self.activation(c_t)
    return h_t, c_t

  def run_all(self, all_input):
    """
    Given all_input, which has size n by 200, where n is the number of
    words, run the forward process n times and return the final h_n.

    TODO: add an option to return the average of all h_i instead of just
    the last one h_n.
    """
    n = all_input.size()[0]
    h_t_tensor = torch.zeros(self.output_dim)
    c_t_tensor = torch.zeros(self.output_dim)
    if torch.cuda.is_available():
        h_t_tensor = h_t_tensor.cuda()
        c_t_tensor = c_t_tensor.cuda()
    h_t = Variable(h_t_tensor)
    c_t = Variable(c_t_tensor)
    for t in xrange(n):
      h_t, c_t = self.forward(all_input[t, :], h_t, c_t)
    return h_t


