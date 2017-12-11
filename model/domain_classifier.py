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
  def __init__(self, input_dim, first_hidden_dim=300, second_hidden_dim=150,
               output_dim=1, use_cuda=False):
    super(DomainClassifier, self).__init__()

    self.fc1 = nn.Linear(input_dim, first_hidden_dim)
    self.activation1 = nn.ReLU()

    self.fc2 = nn.Linear(first_hidden_dim, second_hidden_dim)
    self.activation2 = nn.ReLU()

    # No activation for the third layer since it will be passed through
    # sigmoid in BCEWithLogitsLoss.
    self.fc3 = nn.Linear(second_hidden_dim, output_dim)

  def forward(self, input):
    hidden = self.activation1(self.fc1(input))
    hidden = self.activation2(self.fc2(hidden))
    output = self.fc3(hidden)
    return output.double()

