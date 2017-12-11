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
from lstm_encoder import LSTMEncoder
from domain_classifier import DomainClassifier
from gradient_reversal import GradientReversalLayer

class AdversarialDomainAdaptation(nn.Module):
  def __init__(self, input_dim, cnn_hidden_dim, filter_width, lstm_hidden_dim, Lambda, use_cuda):
    super(AdversarialDomainAdaptation, self).__init__()
    self.question_encoder_cnn = CNNEncoder(input_dim, cnn_hidden_dim, filter_width, use_cuda=use_cuda)
    self.question_encoder_lstm = LSTMEncoder(input_dim, lstm_hidden_dim)
    self.gradient_reversal = GradientReversalLayer(Lambda, use_cuda)
    self.domain_classifier_cnn = DomainClassifier(input_dim=cnn_hidden_dim, use_cuda=use_cuda)
    self.domain_classifier_lstm = DomainClassifier(input_dim=lstm_hidden_dim, use_cuda=use_cuda)

    if use_cuda:
      self.cuda()

  def forward(self, title, body, title_mask, body_mask, use_cnn=True, use_domain_classifider=True, return_average=True):
    """
    Runs one forward pass on the input.

    Return two things:
     - the embedding, so that we can feed it into the loss function for label
       prediction (only if the input came from source not target dataset)
     - the predicted domain label from softmax, so that we can feed it into
       the loss function for domain classification
    """
    title_embedding = None
    body_embedding = None
    if use_cnn:
      title_embedding = self.question_encoder_cnn.run_all(title, title_mask)
      body_embedding = self.question_encoder_cnn.run_all(body, body_mask)
    else:
      title_embedding = self.question_encoder_lstm.run_all(title, title_mask, return_average)
      body_embedding = self.question_encoder_lstm.run_all(body, body_mask, return_average)
    embedding = (title_embedding + body_embedding) / 2
    domain_label = None
    if use_domain_classifider:
      reverse = self.gradient_reversal(embedding)
      if use_cnn:
        domain_label = self.domain_classifier_cnn(reverse)
      else:
        domain_label = self.domain_classifier_lstm(reverse)
    return embedding, domain_label

