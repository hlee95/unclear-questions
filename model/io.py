"""
This file handles loading data, batching data, and creating embeddings.
"""

import sys
import numpy as np
import random

CORPUS_LIMIT = 2000000
TRAINING_EXAMPLE_LIMIT = 200000
EMBEDDINGS_LIMIT = 10000000

EMBEDDING_LENGTH = 200

class Dataset(object):
  def __init__(self):
    # Map id to (title, body).
    self.corpus = {}
    # Map word to 200-dimensional vector.
    self.word_embeddings = {}
    # Array of training examples, 3-tuples (q_i, p_i+, [Q_i-]).
    self.training_examples = []
    # Array of dev/test examples, which are in the form of
    # 3-tuples (query, similar_indexes, candidate_ids).
    self.dev_data = []
    self.test_data = []
    # Keep track of which data sample to return next.
    self.next_training_idx = 0
    self.next_dev_idx = 0
    self.next_test_idx = 0

  def load_corpus(self, filepath):
    print "Loading corpus..."
    corpus_file = open(filepath, "r")
    count = 0
    for line in corpus_file:
      question_id, title, body = line.split("\t")
      self.corpus[int(question_id)] = (title, body)
      count += 1
      if count > CORPUS_LIMIT:
        break
    corpus_file.close()

  def load_vector_embeddings(self, filepath):
    print "Loading vector embeddings..."
    vector_file = open(filepath, "r")
    count = 0
    for line in vector_file:
      tokens = line.split()
      word = tokens[0]
      vector = map(float, tokens[1:])
      self.word_embeddings[word] = np.array(vector)
      count += 1
      if count > EMBEDDINGS_LIMIT:
        break
    vector_file.close()

  def load_training_examples(self, filepath):
    print "Loading training examples..."
    train_file = open(filepath, "r")
    count = 0
    for line in train_file:
      q_i, p_i, Q_i = line.split("\t")
      # For now, only take first p_i if there multiple provided.
      p_i = p_i.split(" ")[0]
      self.training_examples.append((int(q_i), int(p_i), map(int, Q_i.split())))
      count += 1
      if count > TRAINING_EXAMPLE_LIMIT:
        break
    train_file.close()

  def load_dev_data(self, filepath):
    self.dev_data = self.load_eval_data(filepath)

  def load_test_data(self, filepath):
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
      data.append((int(query_id), similar_indexes, candidates))
    return data

  def create_embedding_for_sentence(self, sentence):
    embedding = []
    for word in sentence.split():
      if word in self.word_embeddings:
        embedding.append(self.word_embeddings[word])
      else:
        embedding.append(np.zeros(EMBEDDING_LENGTH))
    return np.array(embedding)

  def get_title(self, id):
    return self.corpus[id][0]

  def get_body(self, id):
    return self.corpus[id][1]

  def get_next_training_feature(self, batch_size=1):
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
      q_i, p_i, Q_i = self.training_examples[self.next_training_idx]
      # Randomly sample 20 negative examples from the 100 given ones.
      negatives = random.sample(xrange(100), 20)
      for sample_id in [q_i, p_i] + [Q_i[j] for j in negatives]:
        embedding = self.create_embedding_for_sentence(self.get_title(sample_id))
        max_n = max(max_n, len(embedding))
        masks.append(np.ones(len(embedding)))
        vectors.append(embedding)
      self.next_training_idx += 1
      self.next_training_idx %= len(self.training_examples)
    # Pad all vectors and masks to the size of the max length one.
    assert len(vectors) == len(masks)
    return self.pad_helper(vectors, masks, batch_size, max_n)

  def get_next_eval_feature(self, use_dev, batch_size=1):
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
      for sample in [query] + candidates:
        embedding = self.create_embedding_for_sentence(self.get_title(sample))
        max_n = max(max_n, len(embedding))
        masks.append(np.ones(len(embedding)))
        vectors.append(embedding)
      if use_dev:
        self.next_dev_idx = (self.next_dev_idx + 1) % len(self.dev_data)
      else:
        self.next_test_idx = (self.next_test_idx + 1) % len(self.test_data)
    padded_vectors, padded_masks = self.pad_helper(vectors, masks, batch_size, max_n)
    return padded_vectors, padded_masks, np.array(similars)

  def pad_helper(self, vectors, masks, batch_size, max_n):
    """
    Helper to pad the vectors and masks arrays when returning features to train
    or evaluate on.
    """
    padded_vectors = np.ndarray((batch_size*22, max_n, EMBEDDING_LENGTH))
    padded_masks = np.ndarray((batch_size*22, max_n))
    for i in xrange(len(vectors)):
      padded_vectors[i] = np.pad(vectors[i], ((0, max_n - len(vectors[i])), (0, 0)), "constant", constant_values=0)
      padded_masks[i] = np.pad(masks[i], (0, max_n - len(masks[i])), "constant", constant_values=0)
    return padded_vectors, padded_masks

