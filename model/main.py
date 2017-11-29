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
NUM_EXAMPLES = 22

USE_CUDA = torch.cuda.is_available()
FLOAT_DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def get_score(q, p):
  score = torch.dot(q, p) / torch.norm(q) / torch.norm(p)
  score = score.data.numpy()[0]
  return score

# TODO: consider using built in function (nn.MultiMarginLoss).
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

def train_lstm(data, lstm, num_epochs, batch_size):
  torch.manual_seed(1)

  for i in range(num_epochs):
    for j in xrange(len(data.training_examples)):
      features, masks = data.get_next_training_feature(batch_size)

      optimizer = optim.Adam(lstm.parameters(), lr=.001, weight_decay=.1)
      optimizer.zero_grad()
      h = lstm.run_all(Variable(torch.Tensor(features).type(FLOAT_DTYPE)))
      loss = Variable(Tensor())
      for k in range(batch_size):
        q_i = h[k*NUM_EXAMPLES, :]
        p_i = h[k*NUM_EXAMPLES + 1, :]
        Q_i = h[k*NUM_EXAMPLES + 2 : (k+1)*NUM_EXAMPLES, :]

        loss += get_loss(h_q, h_p, h_Q)
      # if i%100 == 0:
      #   print i
      #   print loss
      loss.backward()
      optimizer.step()

def train_cnn(data, cnn, num_epochs):
  torch.manual_seed(1)

  for i in range(num_epochs):
    print "Training on %d samples" % len(data.training_examples)
    for j in xrange(len(data.training_examples)):
      features = data.get_next_training_feature()

      q_i = Variable(torch.Tensor(np.expand_dims(features[0].T, 0)).type(FLOAT_DTYPE))
      p_i = Variable(torch.Tensor(np.expand_dims(features[1].T, 0)).type(FLOAT_DTYPE))
      Q_i = features[2:]

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
      if j % 100 == 0:
        print j
        print loss
      loss.backward()
      optimizer.step()

def eval_cnn(data, cnn, use_dev):
  """
  Get evaluation metrics for the CNN. Use dev set if use_dev = True, otherwise
  use test set.
  """
  ranked_scores = []
  for i in xrange(len(data.dev_data)):
    features, similar = data.get_next_eval_feature(use_dev)
    q_i = Variable(torch.Tensor(np.expand_dims(features[0].T, 0)).type(FLOAT_DTYPE))
    # The candidates are everything after the first one.
    C_i = features[1:]
    if USE_CUDA:
      cnn.cuda()
    h_q = torch.squeeze(cnn(q_i), 0)
    candidate_scores = []
    for c in C_i:
      c = Variable(torch.Tensor(np.expand_dims(c.T, 0)).type(FLOAT_DTYPE))
      candidate_scores.append(get_score(h_q, torch.squeeze(cnn(c), 0)))
    # Sort candidate scores in decreasing order and remember which are the
    # correct similar questions.
    ranked_index = np.array(candidate_scores).argsort()
    ranked_score = np.isin(ranked_index, similar).astype(int)
    ranked_scores.append(ranked_score)
  return np.array(ranked_scores)


if __name__ == "__main__":
  data = Dataset()
  data.load_corpus("../data/askubuntu/text_tokenized.txt")
  data.load_vector_embeddings("../data/askubuntu/vector/vectors_pruned.200.txt")
  data.load_training_examples("../data/askubuntu/train_random.txt")
  data.load_dev_data("../data/askubuntu/dev.txt")
  data.load_test_data("../data/askubuntu/test.txt")

  lstm = LSTM(EMBEDDING_LENGTH, HIDDEN_DIM, use_cuda=USE_CUDA)
  if USE_CUDA:
    lstm.cuda()
  train_lstm(data, lstm, 1)

  # cnn = CNN(EMBEDDING_LENGTH, HIDDEN_DIM, FILTER_WIDTH, use_cuda=USE_CUDA)
  # train_cnn(data, cnn, 1)
  # eval_cnn(data, cnn, True)


