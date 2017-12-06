import sys
import numpy as np
import random
# from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import Dataset

TRAINING_EXAMPLE_LIMIT = 200000

class AndroidDataset(Dataset):
  def __init__(self):
    super(AndroidDataset, self).__init__()
    self.positives = {}
    self.negatives = {}

    self.tfidf = None

  def load_training_examples(self, filepath):
    """
    Populate self.training_examples.
    """
    # There aren't any training examples, so we don't bother.
    pass

  def load_dev_data(self, pos_filepath, neg_filepath):
    """
    Populate self.dev_data.
    """
    self.dev_data = self.load_eval_data(pos_filepath, neg_filepath)

  def load_test_data(self, pos_filepath, neg_filepath):
    """
    Populate self.test_data.
    """
    self.test_data = self.load_eval_data(pos_filepath, neg_filepath)

  def load_eval_data(self, pos_filepath, neg_filepath):
    print "Loading eval data from", pos_filepath, "and", neg_filepath
    pos_eval_file = open(pos_filepath, "r")
    data = []
    positives = {}
    negatives = {}
    for line in pos_eval_file:
      query_id, similar_id = map(int, line.split())
      if query_id not in positives:
        positives[query_id] = [similar_id]
      else:
        positives[query_id].append(similar_id)
      # TODO: Check if relationship should be symmetric like this:
      # if similar_id not in positives:
      #   positives[similar_id] = [query_id]
      # else:
      #   positives[similar_id].append(query_id)
    pos_eval_file.close()
    neg_eval_file = open(neg_filepath, "r")
    for line in neg_eval_file:
      query_id, negative_id = map(int, line.split())
      if query_id not in negatives:
        negatives[query_id] = [negative_id]
      else:
        negatives[query_id].append(negative_id)
    neg_eval_file.close()
    for query_id, positive_ids in positives.items():
      candidates = positive_ids + negatives[query_id]
      similar_indexes = range(len(positive_ids))
      assert len(similar_indexes) == len(positive_ids) and len(similar_indexes) > 0
      data.append((query_id, similar_indexes, candidates))
    # TODO: this update might not be completely right, fix it.
    self.positives.update(positives)
    self.negatives.update(negatives)
    return data

  # Overriding.
  def get_next_training_feature_helper(self, batch_size=1, use_title=True):
    """
    Return vectors, which is numpy matrix with dimensions
      batch_size by max_num_words by 200,
    and masks, which is a binary numpy matrix with dimensions
      batch_size by max_num_words.
    """

    # Return embeddings of randomly selected words in the corpus since
    # there aren't any training examples.
    indexes = random.sample(xrange(len(self.corpus.keys())), batch_size)
    ids = [self.corpus.keys()[i] for i in indexes]
    max_n = 0
    vectors = []
    masks = []
    for sample_id in ids:
      embedding = None
      if use_title:
          embedding = self.create_embedding_for_sentence(self.get_title(sample_id))
      else:
        embedding = self.create_embedding_for_sentence(self.get_body(sample_id))
      max_n = max(max_n, len(embedding))
      masks.append(np.ones(len(embedding)))
      vectors.append(embedding)
    # Pad all vectors and masks to the size of the max length one.
    assert len(vectors) == len(masks)
    assert max_n <= self.MAX_SEQUENCE_LENGTH
    return self.pad_helper(vectors, masks, batch_size, max_n)

  # Overriding.
  def get_next_eval_feature_helper(self, use_dev, batch_size=1, use_title=True, use_body=False):
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
    padded_vectors, padded_masks = self.pad_helper(vectors, masks, batch_size, max_n)
    return padded_vectors, padded_masks, np.array(similars)

  def init_tfidf_bow_vectors(self):
    text_only = []
    # for title, body in self.corpus.values():
    #   text_only.append(title)
    #   text_only.append(body)
    # vectorizer = TfidfVectorizer()
    # self.tfidf = vectorizer.fit_transform(corpus)

  def get_next_eval_bow_feature(self, use_dev, batch_size=1):
    """
    Return weighted bag of words vectors for the next batch_size examples.
    The first is the query, then the positives, then the negatives.
    Return the labels as well, which is an array of None for query, 1 for
    positive, and 0 for negative.
    """
    bow_vectors = []
    labels = []
    for _ in xrange(batch_size):
      query, similar, candidates = self.dev_data[self.next_dev_idx] if use_dev else self.test_data[self.next_test_idx]
      labels.append(similar)
      for sample_id in candidates:
        # Get BOW vector.
        bow_vectors.append()

