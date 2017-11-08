from io import Dataset
from io import EMBEDDING_LENGTH
from lstm import LSTM

import sys

import torch
import torch.optim as optim
from torch.autograd import Variable

HIDDEN_DIM = 128
DELTA = 0.00001

USE_CUDA = torch.cuda.is_available()

def get_loss(h_q, h_p, h_Q):
  best_tensor = torch.FloatTensor([-sys.maxint + 1])
  delta_tensor = torch.FloatTensor([DELTA])
  if USE_CUDA:
    best_tensor = best_tensor.cuda()
    delta_tensor = delta_tensor.cuda()
  best = Variable(best_tensor)
  norm_hq = torch.norm(h_q)
  norm_hp = torch.norm(h_p)
  for p in h_Q:
    # compute the score
    val = torch.dot(h_q, p)/(norm_hq * torch.norm(p)) - torch.dot(h_q, h_p)/(norm_hq * norm_hp)
    if val.data[0] > best.data[0]:
      best = val
  # check if p = p+ is the best
  if best.data[0] > DELTA:
    return best
  else:
    return torch.dot(h_q, h_p) - torch.dot(h_q, h_p) + Variable(delta_tensor)

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

  torch.manual_seed(1)

  for i in range(12000):
    features = data.get_next_training_feature()
    q_i_tensor = torch.Tensor(features[0])
    p_i_tensor = torch.Tensor(features[1])

    if USE_CUDA:
      q_i_tensor = q_i_tensor.cuda()
      p_i_tensor = p_i_tensor.cuda()

    q_i = Variable(q_i_tensor)
    p_i = Variable(p_i_tensor)
    Q_i = features[2:]

    lstm = LSTM(EMBEDDING_LENGTH, HIDDEN_DIM)
    if USE_CUDA:
      lstm.cuda()
    optimizer = optim.Adam(lstm.parameters(), lr=.001, weight_decay=.1)
    optimizer.zero_grad()
    h_q = lstm.run_all(q_i)
    h_p = lstm.run_all(p_i)
    h_Q = []
    for q in Q_i:
      q_tensor = torch.Tensor(q)
      if USE_CUDA:
        q_tensor = q_tensor.cuda()
      q = Variable(q_tensor)
      h_Q.append(lstm.run_all(q))
    loss = get_loss(h_q, h_p, h_Q)
    if i%100 == 0:
      print i
      print loss
    loss.backward()
    optimizer.step()


