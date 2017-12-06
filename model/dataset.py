"""
This file implements the base class for Dataset, which handles loading data,
batching data, and creating embeddings.
"""

import sys
import numpy as np
import random

CORPUS_LIMIT = 2000000
EMBEDDINGS_LIMIT = 10000000

EMBEDDING_LENGTH = 200

class Dataset(object):
  def __init__(self):
    self.EMBEDDING_LENGTH = EMBEDDING_LENGTH
    # Trim question title and body to this length.
    self.MAX_SEQUENCE_LENGTH = 50

    # Map id to (title, body).
    self.corpus = {}
    # Store questions we skipped so we exclude them in training/eval.
    self.skipped_questions = {}
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

  def load_training_examples(self, filepath):
    """
    Populate self.training_examples.
    """
    raise NotImplementedError

  def load_dev_data(self, filepath):
    """
    Populate self.dev_data.
    """
    raise NotImplementedError

  def load_test_data(self, filepath):
    """
    Populate self.test_data.
    """
    raise NotImplementedError

  def load_corpus(self, filepath):
    print "Loading corpus..."
    corpus_file = open(filepath, "r")
    count = 0
    for line in corpus_file:
      question_id, title, body = line.split("\t")
      if len(title.split()) == 0 or len(body.split()) == 0:
        print "Skipping question_id %d because title or body is empty" % int(question_id)
        self.skipped_questions[int(question_id)] = True
        continue
      if len(title.split()) > self.MAX_SEQUENCE_LENGTH:
        title = " ".join(title.split()[:self.MAX_SEQUENCE_LENGTH])
      if len(body.split()) > self.MAX_SEQUENCE_LENGTH:
        body = " ".join(body.split()[:self.MAX_SEQUENCE_LENGTH])
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

  def create_embedding_for_sentence(self, sentence):
    embedding = []
    for word in sentence.split():
      if word in self.word_embeddings:
        embedding.append(self.word_embeddings[word])
      else:
        embedding.append(self.word_embeddings["unk"])
    return np.array(embedding)

  def get_title(self, id):
    return self.corpus[id][0]

  def get_body(self, id):
    return self.corpus[id][1]

  def get_next_training_feature(self, batch_size=1, use_title=True, use_body=False):
    title = None
    body= None
    if use_title:
      title_vectors, title_masks = self.get_next_training_feature_helper(batch_size, True)
      title = (title_vectors, title_masks)
    if use_body:
      body_vectors, body_masks = self.get_next_training_feature_helper(batch_size, False)
      body = (body_vectors, body_masks)
    return title, body

  def get_next_training_feature_helper(self, batch_size=1, use_title=True):
    """
    Return vectors, which is numpy matrix with dimensions
      batch_size*22 by max_num_words by 200,
    and masks, which is a binary numpy matrix with dimensions
      batch_size*22 by max_num_words.
    """
    raise NotImplementedError

  def get_next_eval_feature(self, use_dev, batch_size=1, use_title=True, use_body=False):
    title = None
    body = None
    similar = None
    if use_title:
      title_vectors, title_masks, title_similar = \
        self.get_next_eval_feature_helper(use_dev, batch_size, True)
      title = (title_vectors, title_masks)
      similar = title_similar
    if use_body:
      body_vectors, body_masks, body_similar = \
        self.get_next_eval_feature_helper(use_dev_dev, batch_size, False)
      body = (body_vectors, body_masks)
      similar = body_similar
    return title, body, similar

  def get_next_eval_feature_helper(self, use_dev, batch_size=1, use_title=True):
    """
    Returns 3 things:
     - vectors, which is a batch_size*22 by max_n by 200 numpy matrix
     - masks, which is a batch_size*22 by max_n numpy matrix
     - similars, which is a batch_size*22 by 20 matrix of the indexes of the
       samples in candidates that are known to be similar to the query
    """
    raise NotImplementedError

  def pad_helper(self, vectors, masks, batch_size, max_n):
    """
    Helper to pad the vectors and masks arrays when returning features to train
    or evaluate on.
    """
    padded_vectors = np.ndarray((batch_size, max_n, self.EMBEDDING_LENGTH))
    padded_masks = np.ndarray((batch_size, max_n))
    for i in xrange(len(vectors)):
      padded_vectors[i] = np.pad(vectors[i], ((0, max_n - len(vectors[i])), (0, 0)), "constant", constant_values=0)
      padded_masks[i] = np.pad(masks[i], (0, max_n - len(masks[i])), "constant", constant_values=0)
    # Sanity check...
    # for i in xrange(len(padded_vectors)):
    #   for j in xrange(max_n):
    #     # If the embedding at position j is zero, then mask should be 0
    #     if not padded_vectors[i][j].any():
    #       assert padded_masks[i][j] == 0
    #     else:
    #       assert padded_masks[i][j] == 1
    return padded_vectors, padded_masks


