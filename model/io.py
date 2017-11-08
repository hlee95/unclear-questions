"""
This file handles loading data, batching data, and creating embeddings.
"""

import sys
import numpy as np

CORPUS_LIMIT = 2000000
TRAINING_EXAMPLE_LIMIT = 200000

EMBEDDING_LENGTH = 200

class Dataset(object):
  def __init__(self):
    # Map id to (title, body).
    self.corpus = {}
    # Map word to 200-dimensional vector.
    self.word_embeddings = {}
    # Array of training examples, 3-tuples (q_i, p_i+, [Q_i-]).
    self.training_examples = []
    self.next_training_idx = 0

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
    print self.corpus[1]

  def load_vector_embeddings(self, filepath):
    print "Loading vector embeddings..."
    vector_file = open(filepath, "r")
    for line in vector_file:
      tokens = line.split()
      word = tokens[0]
      vector = map(float, tokens[1:])
      self.word_embeddings[word] = np.array(vector)
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

  def get_next_training_feature(self):
    q_i, p_i, Q_i = self.training_examples[self.next_training_idx]
    # Return vectors, which is an array of numpy matrices, where each matrix
    # has dimensions num_words by 200.
    vectors = []
    vectors.append(self.create_embedding_for_sentence(self.get_title(q_i)))
    vectors.append(self.create_embedding_for_sentence(self.get_title(p_i)))
    for q in Q_i:
      vectors.append(self.create_embedding_for_sentence(self.get_title(q)))
    self.next_training_idx += 1
    return vectors


