import sys
import numpy as np
import random

from dataset import Dataset

TRAINING_EXAMPLE_LIMIT = 200000

class AskUbuntuDataset(Dataset):
  def __init__(self):
    super(AskUbuntuDataset, self).__init__()

  def load_training_examples(self, filepath):
    """
    Populate self.training_examples.
    """
    print "Loading training examples..."
    train_file = open(filepath, "r")
    count = 0
    for line in train_file:
      q_i, P_i, Q_i = line.split("\t")
      P_i = map(int, P_i.split())
      Q_i = map(int, Q_i.split())
      self.training_examples.append((int(q_i), P_i, Q_i))
      count += 1
      if count > TRAINING_EXAMPLE_LIMIT:
        break
    train_file.close()

  def load_dev_data(self, filepath):
    """
    Populate self.dev_data.
    """
    self.dev_data = self.load_eval_data(filepath)

  def load_test_data(self, filepath):
    """
    Populate self.test_data.
    """
    self.test_data = self.load_eval_data(filepath)

  def load_eval_data(self, filepath):
    print "Loading eval data from " + filepath
    eval_file = open(filepath, "r")
    data = []
    for line in eval_file:
      query_id, similar_ids, candidate_ids, _ = line.split("\t")
      # Skip entries with no similar ids given.
      if len(similar_ids) == 0:
        continue
      similar = map(int, similar_ids.split())
      candidates = map(int, candidate_ids.split())
      # Find out which candidates are the similar ones.
      similar_indexes = [i for i in xrange(len(candidates)) if candidates[i] in similar]
      assert len(similar_indexes) == len(similar) and len(similar_indexes) > 0
      # Sanity check.
      # for i in xrange(len(candidates)):
      #   if candidates[i] in similar:
      #     assert i in similar_indexes
      #   else:
      #     assert i not in similar_indexes
      data.append((int(query_id), similar_indexes, candidates))
    eval_file.close()
    return data



