from io import Dataset
from io import EMBEDDING_LENGTH
from lstm import LSTM

import torch
import torch.optim as optim
from torch.autograd import Variable

HIDDEN_DIM = 128

def score():
  pass

def get_loss(h_q, h_p, h_Q):
  """
  Return the loss, given the encodings of q, p, and the encodings of
  all the negative examples in Q.
  """
  pass

if __name__ == "__main__":
  data = Dataset()
  data.load_corpus("../data/askubuntu/text_tokenized.txt")
  data.load_vector_embeddings("../data/askubuntu/vector/vectors_pruned.200.txt")
  data.load_training_examples("../data/askubuntu/train_random.txt")
  features = data.get_next_training_feature()

  q_i = Variable(torch.Tensor(features[0]))
  p_i = Variable(torch.Tensor(features[1]))
  Q_i = features[2:]

  lstm = LSTM(EMBEDDING_LENGTH, HIDDEN_DIM)
  optimizer = optim.Adam(lstm.parameters(), lr=.001, weight_decay=.1)
  h_q = lstm.run_all(q_i)
  h_p = lstm.run_all(p_i)
  h_Q = []
  for q in Q_i:
    q = Variable(torch.Tensor(q))
    h_Q.append(lstm.run_all(q))
  loss = get_loss(h_q, h_p, h_Q)


