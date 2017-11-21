from io import Dataset
from io import EMBEDDING_LENGTH
from lstm import LSTM
from cnn import CNN

import sys

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

HIDDEN_DIM = 128
FILTER_WIDTH = 10
DELTA = 0.00001

USE_CUDA = torch.cuda.is_available()
FLOAT_DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def get_loss(h_q, h_p, h_Q):  
  """
  Return the loss, given the encodings of q, p, and the encodings of
  all the negative examples in Q.
  """
  best = Variable(torch.FloatTensor([-sys.maxint + 1]).type(FLOAT_DTYPE))
  norm_hq = torch.norm(h_q)
  norm_hp = torch.norm(h_p)
  for p in h_Q:
    # compute the score
    val = torch.dot(h_q, p)/norm_hq/torch.norm(p) - torch.dot(h_q, h_p)/norm_hq/norm_hp + Variable(torch.FloatTensor([DELTA]).type(FLOAT_DTYPE))
    if val.data[0] > best.data[0]:
      best = val
  # check if p = p+ is the best
  if best.data[0] > DELTA:
    return best
  else:
    return torch.dot(h_q, h_p) - torch.dot(h_q, h_p)

def run_lstm(data, num_iter):
  torch.manual_seed(1)

  for i in range(num_iter):
    features = data.get_next_training_feature()

    q_i = Variable(torch.Tensor(features[0]).type(FLOAT_DTYPE))
    p_i = Variable(torch.Tensor(features[1]).type(FLOAT_DTYPE))
    Q_i = features[2:]

    lstm = LSTM(EMBEDDING_LENGTH, HIDDEN_DIM, use_cuda=USE_CUDA)
    if USE_CUDA:
      lstm.cuda()
    optimizer = optim.Adam(lstm.parameters(), lr=.001, weight_decay=.1)
    optimizer.zero_grad()
    h_q = lstm.run_all(q_i)
    h_p = lstm.run_all(p_i)
    h_Q = []
    for q in Q_i:
      q = Variable(torch.Tensor(q).type(FLOAT_DTYPE))
      h_Q.append(lstm.run_all(q))
    loss = get_loss(h_q, h_p, h_Q)
    if i%100 == 0:
      print i
      print loss
    loss.backward()
    optimizer.step()

def run_cnn(data, num_iter):
  torch.manual_seed(1)

  for i in range(num_iter):
    features = data.get_next_training_feature()

    q_i = Variable(torch.Tensor(np.expand_dims(features[0].T, 0)).type(FLOAT_DTYPE))
    p_i = Variable(torch.Tensor(np.expand_dims(features[1].T, 0)).type(FLOAT_DTYPE))
    Q_i = features[2:]

    cnn = CNN(EMBEDDING_LENGTH, HIDDEN_DIM, FILTER_WIDTH, use_cuda=USE_CUDA)
    if USE_CUDA:
      cnn.cuda()
    optimizer = optim.Adam(cnn.parameters(), lr=.001, weight_decay=.1)
    optimizer.zero_grad()
    h_q = torch.squeeze(cnn(q_i), 0)
    h_p = torch.squeeze(cnn(p_i), 0)
    h_Q = []
    for q in Q_i:
      q = Variable(torch.Tensor(np.expand_dims(q.T, 0)).type(FLOAT_DTYPE))
      h_Q.append(torch.squeeze(cnn(q), 0))
    loss = get_loss(h_q, h_p, h_Q)
    print i
    print loss
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
  data = Dataset()
  data.load_corpus("../data/askubuntu/text_tokenized.txt")
  data.load_vector_embeddings("../data/askubuntu/vector/vectors_pruned.200.txt")
  data.load_training_examples("../data/askubuntu/train_random.txt")

  run_cnn(data, 100)

