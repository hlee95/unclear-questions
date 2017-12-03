from dataset import Dataset
from dataset import EMBEDDING_LENGTH
from askubuntu_dataset import AskUbuntuDataset
from android_dataset import AndroidDataset
from lstm_encoder import LSTMEncoder
from cnn_encoder import CNNEncoder
from eval import Eval

import sys

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

CNN_HIDDEN_DIM = 667
LSTM_HIDDEN_DIM = 128
FILTER_WIDTH = 10
DELTA = 0.2
NUM_EXAMPLES = 22
LR = 0.0001
WD = 0.001

USE_CUDA = torch.cuda.is_available()
FLOAT_DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def get_score(q, p):
  """
  Returns the cosine similarity between q and p, which are both expected to be
  tensors of size (n,) for some n.
  """
  score = torch.dot(q, p) / torch.norm(q) / torch.norm(p)
  score = score.cpu().data.numpy()[0]
  return score

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
    val = torch.dot(h_q, p)/norm_hq/torch.norm(p) - \
          torch.dot(h_q, h_p)/norm_hq/norm_hp + \
          Variable(torch.FloatTensor([DELTA]).type(FLOAT_DTYPE))
    if val.data[0] > best.data[0]:
      best = val
  # check if p = p+ is the best
  if best.data[0] > 0:
    return best
  else:
    return torch.dot(h_q, h_p) - torch.dot(h_q, h_p)

def train_lstm(data, lstm, num_epochs, batch_size):
  torch.manual_seed(1)
  optimizer = optim.Adam(lstm.parameters(), lr=LR, weight_decay=WD)
  print "Training LSTM on %d samples..." % len(data.training_examples)
  for i in range(num_epochs):
    print "==================\nEpoch: %d of %d\n==================" % (i + 1, num_epochs)
    num_batches = len(data.training_examples)/batch_size
    print "num_batches", num_batches
    for j in xrange(num_batches):
      features, masks = data.get_next_training_feature(batch_size)

      optimizer.zero_grad()
      h = lstm.run_all(
        Variable(torch.Tensor(features).type(FLOAT_DTYPE)),
        Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
      )

      for k in range(batch_size):
        h_q = h[k*NUM_EXAMPLES, :]
        h_p = h[k*NUM_EXAMPLES + 1, :]
        h_Q = h[k*NUM_EXAMPLES + 2 : (k+1)*NUM_EXAMPLES, :]

        loss = get_loss(h_q, h_p, h_Q)
        loss.backward(retain_graph=True)

      optimizer.step()
      if j % (250) == 0:
        print "batch number", j

def eval_lstm(data, lstm, use_dev):
  print "Evaluating LSTM..."
  ranked_scores = []
  for i in xrange(len(data.dev_data)):
    features, masks, similar = data.get_next_eval_feature(use_dev)
    h = lstm.run_all(
      Variable(torch.Tensor(features).type(FLOAT_DTYPE)),
      Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
    )
    candidate_scores = []
    # The candidates are all results after the first one, which is h_q.
    h_q = h[0]
    for c in h[1:]:
      candidate_scores.append(get_score(h_q, c))
    # Sort candidate scores in decreasing order and remember which are the
    # correct similar questions.
    ranked_index = np.array(candidate_scores).argsort()
    ranked_score = np.isin(ranked_index, similar).astype(int)
    ranked_scores.append(ranked_score)
  lstm_eval = Eval(np.array(ranked_scores))
  print "MAP:", lstm_eval.MAP()
  print "MRR:", lstm_eval.MRR()
  print "Precision@1:", lstm_eval.Precision(1)
  print "Precision@5:", lstm_eval.Precision(5)
  return np.array(ranked_scores)
  return np.array(ranked_scores)

def train_cnn(data, cnn, num_epochs, batch_size):
  torch.manual_seed(1)
  optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=WD)
  print "Training CNN on %d samples..." % len(data.training_examples)
  for i in range(num_epochs):
    print "==================\nEpoch: %d of %d\n==================" % (i + 1, num_epochs)
    num_batches = len(data.training_examples)/batch_size
    print "num_batches", num_batches
    for j in xrange(num_batches):
      features, masks = data.get_next_training_feature(batch_size)
      features_T = np.swapaxes(features, 1, 2)

      optimizer.zero_grad()
      h = cnn(
        Variable(torch.Tensor(features_T).type(FLOAT_DTYPE)),
        Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
      )

      for k in range(batch_size):
        h_q = h[k*NUM_EXAMPLES, :]
        h_p = h[k*NUM_EXAMPLES + 1, :]
        h_Q = h[k*NUM_EXAMPLES + 2 : (k+1)*NUM_EXAMPLES, :]

        loss = get_loss(h_q, h_p, h_Q)
        loss.backward(retain_graph=True)

      if j % (100) == 0:
        print "batch number", j
      optimizer.step()
    eval_cnn(data, cnn, True)


def eval_cnn(data, cnn, use_dev):
  """
  Get evaluation metrics for the CNN. Use dev set if use_dev = True, otherwise
  use test set.
  """
  print "Evaluating CNN..."
  ranked_scores = []
  for i in xrange(len(data.dev_data)):
    features, masks, similar = data.get_next_eval_feature(use_dev)
    features_T = np.swapaxes(features, 1, 2)
    h = cnn(Variable(torch.Tensor(features_T).type(FLOAT_DTYPE)),
            Variable(torch.Tensor(masks).type(FLOAT_DTYPE)))
    candidate_scores = []
    # The candidates are all results after the first one, which is h_q.
    h_q = h[0]
    for c in h[1:]:
      candidate_scores.append(get_score(h_q, c))
    # Sort candidate scores in decreasing order and remember which are the
    # correct similar questions.
    ranked_index = np.array(candidate_scores).argsort()
    ranked_score = np.isin(ranked_index, similar).astype(int)
    ranked_scores.append(ranked_score)
  cnn_eval = Eval(np.array(ranked_scores))
  print "MAP:", cnn_eval.MAP()
  print "MRR:", cnn_eval.MRR()
  print "Precision@1:", cnn_eval.Precision(1)
  print "Precision@5:", cnn_eval.Precision(5)
  return np.array(ranked_scores)


def part1(askubuntu_data, mode):
  if mode == 'lstm':
    lstm = LSTMEncoder(EMBEDDING_LENGTH, LSTM_HIDDEN_DIM, use_cuda=USE_CUDA)
    train_lstm(askubuntu_data, lstm, 1, 5)

  if mode == 'cnn':
    cnn = CNNEncoder(EMBEDDING_LENGTH, CNN_HIDDEN_DIM, FILTER_WIDTH, use_cuda=USE_CUDA)
    train_cnn(askubuntu_data, cnn, 3, 5)

def part2(askubuntu_data, android_data):
  # TODO: Train and evaluate the adversarial domain adapatation network.
  pass

if __name__ == "__main__":
  if USE_CUDA:
    print "using CUDA"
  # Load all the data!
  askubuntu_data = AskUbuntuDataset()
  askubuntu_data.load_corpus("../data/askubuntu/text_tokenized.txt")
  askubuntu_data.load_vector_embeddings("../data/askubuntu/vector/vectors_pruned.200.txt")
  askubuntu_data.load_training_examples("../data/askubuntu/train_random.txt")
  askubuntu_data.load_dev_data("../data/askubuntu/dev.txt")
  askubuntu_data.load_test_data("../data/askubuntu/test.txt")

  # android_data = AndroidDataset()
  # android_data.load_corpus("../data/android/corpus.tsv")
  # TODO: Load vector embeddings from glove not from the askubuntu vectors pruned.
  # android_data.load_vector_embeddings("../data/askubuntu/vector/vectors_pruned.200.txt")
  # android_data.load_dev_data("../data/android/dev.pos.txt", "../data/android/dev.neg.txt")
  # android_data.load_test_data("../data/android/test.pos.txt", "../data/android/test.neg.txt")

  part1(askubuntu_data, mode='cnn')


