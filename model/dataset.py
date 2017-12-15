"""
This file implements the base class for Dataset, which handles loading data,
batching data, and creating embeddings.
"""

import sys
import numpy as np
import random

CORPUS_LIMIT = 2000000
EMBEDDINGS_LIMIT = 10000000

EMBEDDING_LENGTH = 300

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
    self.tfidf_dicts = {}


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
      title_words = [word.lower() for word in title.split()]
      body_words = [word.lower() for word in body.split()]
      if len(title_words) > self.MAX_SEQUENCE_LENGTH:
        title_words = title_words[:self.MAX_SEQUENCE_LENGTH]
      if len(body_words) > self.MAX_SEQUENCE_LENGTH:
        body_words = body_words[:self.MAX_SEQUENCE_LENGTH]
      self.corpus[int(question_id)] = (" ".join(title_words), " ".join(body_words))
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

  def get_next_training_feature(self, batch_size=1, use_title=True, use_body=True, tfidf_weighting=False):
    title = None
    body= None
    if use_title:
      title_vectors, title_masks = self.get_next_training_feature_helper(batch_size, True, tfidf_weighting)
      title = (title_vectors, title_masks)
    if use_body:
      if use_title:
        # If we already got title features, backtrack so we get the same features.
        if len(self.training_examples) > 0:
          self.next_training_idx = (self.next_training_idx - batch_size) % len(self.training_examples)
      body_vectors, body_masks = self.get_next_training_feature_helper(batch_size, False, tfidf_weighting)
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

  def get_next_eval_feature(self, use_dev, batch_size=1, use_title=True, use_body=True, tfidf_weighting=False):
    title = None
    body = None
    similar = None

    if use_title:
      title_vectors, title_masks, title_similar = \
        self.get_next_eval_feature_helper(use_dev, batch_size, True, tfidf_weighting)
      title = (title_vectors, title_masks)
      similar = title_similar

    if use_body:
      if use_title:
        # If we already got title features, backtrack so we get the same features.
        if use_dev:
          self.next_dev_idx = (self.next_dev_idx - batch_size) % len(self.dev_data)
        else:
          self.next_test_idx = (self.next_test_idx - batch_size) % len(self.test_data)
      body_vectors, body_masks, body_similar = \
        self.get_next_eval_feature_helper(use_dev, batch_size, False, tfidf_weighting)
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

  def init_tfidf_bow_vectors(self):
    # find vocab
    # find df
    # find tf
    print "Computing Tf-Idf vectors..."
    vocab_dict = {}
    vocab_list = []
    document_frequency = {}
    for key in self.corpus:
      text_only = self.corpus[key][0].split() + self.corpus[key][1].split()
      for word in np.unique(text_only):
        if not word in vocab_dict:
          vocab_dict[word] = len(vocab_list)-1
          vocab_list.append(word)
        document_frequency[word] = document_frequency.get(word, 0) + 1

    num_docs = len(self.corpus)
    for key in self.corpus:
      text_only = self.corpus[key][0].split() + self.corpus[key][1].split()
      unique, counts = np.unique(text_only, return_counts=True)
      # normalized_counts = counts / len(text_only)
      tfidf_dict = {}
      for i in range(len(unique)):
        word = unique[i]
        tfidf = np.log(float(num_docs) / document_frequency[word]) * counts[i] / len(text_only)
        tfidf_dict[word] = tfidf
      self.tfidf_dicts[key] = tfidf_dict

  def get_bow_feature(self, sample_id):
    title = self.get_title(sample_id).split()
    body = self.get_body(sample_id).split()
    bow_title = np.zeros(len(title))
    bow_body = np.zeros(len(body))
    d = self.tfidf_dicts[sample_id]
    for i in range(len(title)):
      word = title[i]
      bow_title[i] = d[word]
    for i in range(len(body)):
      word = body[i]
      bow_body[i] = d[word]
    return (bow_title, bow_body)

