"""
This file implements the domain classifier for Part 2 of the project.

The domain classifier is a feedforward network with 3 fully connected layers.
The first two use a ReLU activation, and the third one uses logistic activation
(which is just a softmax with only 2 outputs).

The design is from Figure 5 in the paper "Unsupervised Domain Adaptation by
Backpropagation."
"""
import torch
from torch import nn
from torch.autograd import Variable

class DomainClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim=2, use_cuda=False):
    super(DomainClassifier, self).__init__()

    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.activation1 = nn.ReLU()

    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.activation2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_dim, output_dim)
    self.activation3 = nn.LogSoftmax()

  def forward(self, input):
    hidden = self.activation1(self.fc1(input))
    hidden = self.activation2(self.fc2(hidden))
    output = self.activation3(self.fc3(hidden))
    return output

