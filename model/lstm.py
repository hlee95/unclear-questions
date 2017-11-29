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
    print "sizes:"
    print input.size()
    print prev_hidden.size()
    hidden = self.i2h(torch.cat((input, prev_hidden), 1))
    hidden = self.h_activation(hidden)
    return hidden

class LSTM(nn.Module):
  def __init__(self, input_dim, output_dim, activation=nn.Tanh(), use_cuda=False):
    super(LSTM, self).__init__()
    # Create the different gates that we need.
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.i_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.f_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.o_gate = RNNLayer(input_dim, output_dim, nn.Sigmoid())
    self.z = RNNLayer(input_dim, output_dim, nn.Tanh())
    self.activation = activation
    self.float_dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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

  def run_all(self, all_input, mask=None, return_average=True):
    """
    Given all_input, which has shape batch_size by max_num_words by 200, and a
    mask with shape batch_size by max_num_words, indicating the true length of
    each of the batch_size inputs, run the forward process max_num_words times
    and return the final h_n of each of the inputs (using the mask to ignore
    extra padding in some inputs).

    TODO: add an option to return the average of all h_i instead of just
    the last one h_n.
    """
    # broadcasted_mask = np.repeat(mask, self.output_dim, 2)
    batch_size = all_input.size()[0]
    max_num_words = all_input.size()[1]
    # h_t = Variable(torch.zeros(batch_size, self.output_dim).type(self.float_dtype))
    # c_t = Variable(torch.zeros(batch_size, self.output_dim).type(self.float_dtype))
    h = Variable(torch.zeros(batch_size, max_num_words, self.output_dim).type(self.float_dtype))
    c = Variable(torch.zeros(batch_size, max_num_words, self.output_dim).type(self.float_dtype))
    for t in xrange(max_num_words):
      h[:,t,:], c[:,t,:] = self.forward(all_input[:, t, :], h[:,t-1,:], c[:,t-1,:])
    masked_h = h*mask.unsqueeze(2)

    if return_average:
        return torch.sum(masked_h, 1)/torch.sum(mask).unsqueeze(0)
    else:
        # TODO: take into account how long the sentence is by using mask
        return masked_h[:,-1,:]


