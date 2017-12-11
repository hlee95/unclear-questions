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

    self.tfidf_dicts = {}

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
    # Merge with global positive and negative dictionaries.
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
  def get_next_eval_feature_helper(self, use_dev, batch_size=1, use_title=True):
    """
    Returns 3 things:
     - vectors, which is a batch_size*21 by max_n by 200 numpy matrix
     - masks, which is a batch_size*21 by max_n numpy matrix
     - similars, which is a batch_size*21 by 20 matrix of the indexes of the
       samples in candidates that are known to be similar to the query
    """
    max_n = 0
    vectors = []
    similars = []
    masks = []
    num_candidates = 20 # Some positive, some negative.
    for _ in xrange(batch_size):
      query, similar, candidates = self.dev_data[self.next_dev_idx] if use_dev else self.test_data[self.next_test_idx]
      similars.append(similar)
      random_negative_idxs = random.sample(xrange(len(similar), len(candidates)), num_candidates - len(similar))
      # Sanity check.
      # for i in xrange(len(similar)):
      #   assert i in similar
      #   assert i not in random_negative_idxs
      shorter_candidates_list = [candidates[i] for i in xrange(len(candidates)) if i in similar or i in random_negative_idxs]
      for sample_id in [query] + shorter_candidates_list:
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
    padded_vectors, padded_masks = self.pad_helper(vectors, masks, batch_size*21, max_n)
    return padded_vectors, padded_masks, np.array(similars)

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

  def get_next_eval_bow_feature(self, use_dev, batch_size=1):
    """
    Return weighted bag of words vectors for the next batch_size examples.
    The first is the query, then the positives, then the negatives.
    Return the labels as well, which is an array of None for query, 1 for
    positive, and 0 for negative.
    """
    bow_vectors = []
    labels = []
    num_negatives = 20
    for _ in xrange(batch_size):
      query, similar, candidates = self.dev_data[self.next_dev_idx] if use_dev else self.test_data[self.next_test_idx]
      random_negative_idxs = random.sample(xrange(len(similar), len(candidates)), num_negatives)
      for i in xrange(len(similar)):
        assert i in similar
        assert i not in random_negative_idxs
      labels_batch = np.zeros(len(similar) + num_negatives)
      for i in similar:
        labels_batch[i] = 1
      labels.append(labels_batch)
      shorter_candidates_list = [candidates[i] for i in xrange(len(candidates)) if i in similar or i in random_negative_idxs]
      for sample_id in [query] + shorter_candidates_list:
        # Get BOW vector.
        bow_vectors.append(self.tfidf_dicts[sample_id])
      if use_dev:
        self.next_dev_idx = (self.next_dev_idx + 1) % len(self.dev_data)
      else:
        self.next_test_idx = (self.next_test_idx + 1) % len(self.test_data)
    return (bow_vectors, labels)

