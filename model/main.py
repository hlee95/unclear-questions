from dataset import Dataset
from dataset import EMBEDDING_LENGTH
from askubuntu_dataset import AskUbuntuDataset
from android_dataset import AndroidDataset
from lstm_encoder import LSTMEncoder
from cnn_encoder import CNNEncoder
from eval import Eval

import sys
from enum import Enum

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

CNN_HIDDEN_DIM = 667
LSTM_HIDDEN_DIM = 128
FILTER_WIDTH = 10
DELTA = 0.2
NUM_EXAMPLES = 22
LR = 0.001
WD = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 50

USE_CUDA = torch.cuda.is_available()
FLOAT_DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class ModelType(Enum):
  CNN = 0
  LSTM = 1

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

def run_model(model, title, body, use_title, use_body, model_type):
  """
  Return the encodings of the title, the body, or the average,
  depending on if use_title and use_body are True.

  Parameter model_type is either "lstm" or "cnn"
  """
  assert type(model_type) == ModelType
  assert use_title or use_body
  h = None
  # Case where we only use title encodings.
  if use_title and not use_body:
    features, masks = title[0], title[1]
    if model_type == ModelType.CNN:
      features = np.swapaxes(features, 1, 2)
    h = model.run_all(
      Variable(torch.Tensor(features).type(FLOAT_DTYPE)),
      Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
    )
  # Case where we only use body encodings.
  if use_body and not use_title:
    features, masks = body[0], body[1]
    if model_type == ModelType.CNN:
      features = np.swapaxes(features, 1, 2)
    h = model.run_all(
      Variable(torch.Tensor(features).type(FLOAT_DTYPE)),
      Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
    )
  # Case where we average the title and body encodings.
  if use_body and use_title:
    title_features, title_masks = title[0], title[1]
    if model_type == ModelType.CNN:
      title_features = np.swapaxes(title_features, 1, 2)
    h_title = model.run_all(
      Variable(torch.Tensor(title_features).type(FLOAT_DTYPE)),
      Variable(torch.Tensor(title_masks).type(FLOAT_DTYPE))
    )
    body_features, body_masks = body[0], body[1]
    if model_type == ModelType.CNN:
      body_features = np.swapaxes(body_features, 1, 2)
    h_body = model.run_all(
      Variable(torch.Tensor(body_features).type(FLOAT_DTYPE)),
      Variable(torch.Tensor(body_masks).type(FLOAT_DTYPE))
    )
    h = (h_title + h_body) / 2
  return h

def train_model(model_type, data, model, num_epochs, batch_size, use_title=True, use_body=False):
  """
  Train the given model with the given data.
  """
  torch.manual_seed(1)
  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
  print "Training %s on %d samples..." % (model_type.name, len(data.training_examples))
  for i in range(num_epochs):
    print "==================\nEpoch: %d of %d\n==================" % (i + 1, num_epochs)
    num_batches = len(data.training_examples)/batch_size
    print "num_batches", num_batches
    for j in xrange(num_batches):
      title, body = data.get_next_training_feature(batch_size, use_title, use_body)
      optimizer.zero_grad()
      h = run_model(model, title, body, use_title, use_body, model_type)
      avg_loss = 0
      for k in range(batch_size):
        h_q = h[k*NUM_EXAMPLES, :]
        h_p = h[k*NUM_EXAMPLES + 1, :]
        h_Q = h[k*NUM_EXAMPLES + 2 : (k+1)*NUM_EXAMPLES, :]

        loss = get_loss(h_q, h_p, h_Q)
        avg_loss += loss.data[0]
        loss.backward(retain_graph=True)
      avg_loss /= batch_size
      # print avg_loss
      optimizer.step()
      if j % (250) == 0:
        print "batch number %d, loss %f" % (j, avg_loss)
    eval_model(model, data, model_type, False)
    eval_model(model, data, model_type, True)

def eval_model(model, data, model_type, use_dev, use_title=True, use_body=False):
  print "Evaluating %s on %s dataset..." % (model_type.name, 'dev' if use_dev else 'train')
  ranked_scores = []
  for i in xrange(len(data.dev_data)):
    title, body, similar = data.get_next_eval_feature(use_dev)
    h = run_model(model, title, body, use_title, use_body, model_type)
    candidate_scores = []
    # The candidates are all results after the first one, which is h_q.
    h_q = h[0]
    for c in h[1:]:
      candidate_scores.append(get_score(h_q, c))
    # Sort candidate scores in decreasing order and remember which are the
    # correct similar questions.
    ranked_index = np.array(candidate_scores).argsort()[::-1]
    ranked_score = np.isin(ranked_index, similar).astype(int)
    ranked_scores.append(ranked_score)
  eval_obj = Eval(np.array(ranked_scores))
  print "MAP:", eval_obj.MAP()
  print "MRR:", eval_obj.MRR()
  print "Precision@1:", eval_obj.Precision(1)
  print "Precision@5:", eval_obj.Precision(5)

def part1(askubuntu_data, mode):
  if mode == ModelType.LSTM:
    lstm = LSTMEncoder(EMBEDDING_LENGTH, LSTM_HIDDEN_DIM, use_cuda=USE_CUDA)
    train_model(mode, askubuntu_data, lstm, NUM_EPOCHS, BATCH_SIZE, use_title=True, use_body=False)

  if mode == ModelType.CNN:
    cnn = CNNEncoder(EMBEDDING_LENGTH, CNN_HIDDEN_DIM, FILTER_WIDTH, use_cuda=USE_CUDA)
    train_model(mode, askubuntu_data, cnn, NUM_EPOCHS, BATCH_SIZE, use_title=True, use_body=False)

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
  # android_data.load_vector_embeddings("../data/glove/glove_pruned_200D.txt")
  # android_data.load_dev_data("../data/android/dev.pos.txt", "../data/android/dev.neg.txt")
  # android_data.load_test_data("../data/android/test.pos.txt", "../data/android/test.neg.txt")

  part1(askubuntu_data, mode=ModelType.LSTM)

