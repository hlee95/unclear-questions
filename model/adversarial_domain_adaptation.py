"""
This file implements the entire network for Part 2 of the project.

This network is composed of multiple modules:
 - the question encoder (feature extractor) (from Part 1)
 - the domain classifier
 - the label predictor

We define a custom backward() function to implement the gradient reversal
specified in the paper.
"""
import torch
from torch import nn
from torch.autograd import Variable

from cnn_encoder import CNNEncoder
from domain_classifier import DomainClassifier

class AdversarialDomainAdaptation(nn.Module):
  def __init__(self, input_dim, qe_hidden_dim, filter_width, dc_hidden_dim, Lambda, use_cuda):
    super(AdversarialDomainAdaptation, self).__init__()
    self.question_encoder = CNNEncoder(input_dim, qe_hidden_dim, filter_width, use_cuda=use_cuda)
    self.domain_classifier = DomainClassifier(qe_hidden_dim, dc_hidden_dim, use_cuda=use_cuda)
    self.Lambda = Lambda # Capitalized because lambda is a Python keyword.

    # TODO: Save parameters for later use in backward() function.
    self.parameters =[]

    if use_cuda:
      self.cuda()

  def forward(self, input, mask=None, return_average=True):
    """
    Runs one forward pass on the input.

    Return two things:
     - the embedding, so that we can feed it into the loss function for label
       prediction (only if the input came from source not target dataset)
     - the predicted domain label from softmax, so that we can feed it into
       the loss function for domain classification
    """
    embedding = self.question_encoder(input, mask, return_average)
    domain_label = self.domain_classifier(embedding)
    return embedding, domain_label

  def backward(self, grad_output):
    pass
    # TODO: We need to implement this, maybe this is helpful:
    # https://discuss.pytorch.org/t/defining-backward-function-in-nn-module/5047/2

