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
      # for x in map(int, P_i.split() + Q_i.split()):
      #   if x in self.skipped_questions:
      #     print "in training"
      P_i = filter(
        lambda x: x not in self.skipped_questions,
        map(int, P_i.split())
      )
      Q_i = filter(
        lambda x: x not in self.skipped_questions,
        map(int, Q_i.split())
      )
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
      # for x in map(int, similar_ids.split() + candidate_ids.split()):
      #   if x in self.skipped_questions:
      #     print "in eval"
      similar = filter(
        lambda x: x not in self.skipped_questions,
        map(int, similar_ids.split())
      )
      candidates = filter(
        lambda x: x not in self.skipped_questions,
        map(int, candidate_ids.split())
      )
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

  # Overriding.
  def get_next_training_feature_helper(self, batch_size=1, use_title=True):
    """
    Return vectors, which is numpy matrix with dimensions
      batch_size*22 by max_num_words by 200,
    and masks, which is a binary numpy matrix with dimensions
      batch_size*22 by max_num_words.
    """
    max_n = 0
    vectors = []
    masks = []
    for _ in xrange(batch_size):
      q_i, P_i, Q_i = self.training_examples[self.next_training_idx]
      # Randomly select a positive example p_i from P_i.
      positive = random.randint(0, len(P_i) - 1)
      p_i = P_i[positive]
      # Randomly sample 20 negative examples from the 100 given ones.
      negatives = random.sample(xrange(len(Q_i)), 20)
      for sample_id in [q_i, p_i] + [Q_i[j] for j in negatives]:
        embedding = None
        if use_title:
          embedding = self.create_embedding_for_sentence(self.get_title(sample_id))
        else:
          embedding = self.create_embedding_for_sentence(self.get_body(sample_id))
        max_n = max(max_n, len(embedding))
        masks.append(np.ones(len(embedding)))
        vectors.append(embedding)
      self.next_training_idx += 1
      self.next_training_idx %= len(self.training_examples)
    # Pad all vectors and masks to the size of the max length one.
    assert len(vectors) == len(masks)
    assert max_n <= self.MAX_SEQUENCE_LENGTH
    return self.pad_helper(vectors, masks, batch_size * 22, max_n)

  # Overriding.
  def get_next_eval_feature_helper(self, use_dev, batch_size=1, use_title=True):
    """
    Returns 3 things:
     - vectors, which is a batch_size*22 by max_n by 200 numpy matrix
     - masks, which is a batch_size*22 by max_n numpy matrix
     - similars, which is a batch_size*22 by 20 matrix of the indexes of the
       samples in candidates that are known to be similar to the query
    """
    max_n = 0
    vectors = []
    similars = []
    masks = []
    for _ in xrange(batch_size):
      query, similar, candidates = self.dev_data[self.next_dev_idx] if use_dev else self.test_data[self.next_test_idx]
      similars.append(similar)
      for sample_id in [query] + candidates:
        embedding = None
        if use_title:
          embedding = self.create_embedding_for_sentence(self.get_title(sample_id))
        else:
          embedding = self.create_embedding_for_sentence(self.get_body(sample_id))
        max_n = max(max_n, len(embedding))
        masks.append(np.ones(len(embedding)))
        vectors.append(embedding)
      if use_dev:
        self.next_dev_idx = (self.next_dev_idx + 1) % len(self.dev_data)
      else:
        self.next_test_idx = (self.next_test_idx + 1) % len(self.test_data)
    assert max_n <= self.MAX_SEQUENCE_LENGTH
    padded_vectors, padded_masks = self.pad_helper(vectors, masks, batch_size * 22, max_n)
    return padded_vectors, padded_masks, np.array(similars)
