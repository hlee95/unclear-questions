from dataset import Dataset
from dataset import EMBEDDING_LENGTH
from askubuntu_dataset import AskUbuntuDataset
from android_dataset import AndroidDataset
from lstm_encoder import LSTMEncoder
from cnn_encoder import CNNEncoder
from eval import Eval
from meter import AUCMeter
from adversarial_domain_adaptation import AdversarialDomainAdaptation
from gradient_reversal import GradientReversalLayer
import sys
from enum import Enum

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import time

CNN_HIDDEN_DIM = 667
LSTM_HIDDEN_DIM = 128
LAMBDA = 0.1
FILTER_WIDTH = 3
DELTA = 0.2
NUM_EXAMPLES = 22
LR = 0.0001
WD = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 1

USE_CUDA = torch.cuda.is_available()
FLOAT_DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DOUBLE_DTYPE = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor

class ModelType(Enum):
  CNN = 0
  LSTM = 1

def get_cosine_similarity(q, p):
  """
  Returns the cosine similarity between q and p, which are both expected to be
  tensors of size (n,) for some n.
  """
  score = torch.dot(q, p) / torch.norm(q) / torch.norm(p)
  score = score.cpu().data.numpy()[0]
  return score

def get_tfidf_cosine_similarity(d1, d2):
  d1_norm_squared = 0
  d2_norm_squared = 0
  dot_product = 0
  for word, val in d1.items():
    dot_product += val * d2.get(word, 0)
    d1_norm_squared += val*val
  for val in d2.values():
    d2_norm_squared += val*val
  return dot_product/np.sqrt(d1_norm_squared*d2_norm_squared)

def get_multimargin_loss(h_q, h_p, h_Q):
  """
  Return the loss, given the encodings of q, p, and the encodings of
  all the negative examples in Q.
  """
  best = Variable(torch.FloatTensor([-sys.maxint + 1]).type(FLOAT_DTYPE))
  norm_hq = torch.norm(h_q)
  norm_hp = torch.norm(h_p)
  for p in h_Q:
    # compute the score
    val = torch.dot(h_q, p)/norm_hq/torch.norm(p) - \
          torch.dot(h_q, h_p)/norm_hq/norm_hp + \
          Variable(torch.FloatTensor([DELTA]).type(FLOAT_DTYPE))
    if val.data[0] > best.data[0]:
      best = val
  # check if p = p+ is the best
  if best.data[0] > 0:
    return best
  else:
    return torch.dot(h_q, h_p) - torch.dot(h_q, h_p)

def run_model(model, title, body, use_title, use_body, model_type):
  """
  Return the encodings of the title, the body, or the average,
  depending on if use_title and use_body are True.

  Parameter model_type is either "lstm" or "cnn"
  """
  assert type(model_type) == ModelType
  assert use_title or use_body
  h = None
  if use_body and use_title:
    # Case where we average the title and body encodings.
    title_features, title_masks = title[0], title[1]
    h_title = run_model_helper(model, title_features, title_masks, model_type)
    body_features, body_masks = body[0], body[1]
    h_body = run_model_helper(model, body_features, body_masks, model_type)
    h = (h_title + h_body) / 2
  else:
    # Case where we only use either title or body encodings.
    features, masks = None, None
    if use_title:
      features, masks = title[0], title[1]
    else:
      features, masks = body[0], body[1]
    h = run_model_helper(model, features, masks, model_type)
  return h

def run_model_helper(model, features, masks, model_type):
  if model_type == ModelType.CNN:
    features = np.swapaxes(features, 1, 2)
  return model.run_all(
    Variable(torch.Tensor(features).type(FLOAT_DTYPE)),
    Variable(torch.Tensor(masks).type(FLOAT_DTYPE))
  )

def train_model(model_type, data, model, num_epochs, batch_size, use_title=True,
                use_body=True, tfidf_weighting=False):
  """
  Train the given model with the given data.
  """
  torch.manual_seed(1)
  start = time.time()
  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
  print "Training %s on %d samples..." % (model_type.name, len(data.training_examples))
  for i in range(num_epochs):
    print "==================\nEpoch: %d of %d\n==================" % (i + 1, num_epochs)
    num_batches = len(data.training_examples)/batch_size
    print "num_batches", num_batches
    for j in xrange(num_batches):
      title, body = data.get_next_training_feature(batch_size, use_title,
        use_body, tfidf_weighting)
      optimizer.zero_grad()
      h = run_model(model, title, body, use_title, use_body, model_type)
      avg_loss = 0
      for k in range(batch_size):
        h_q = h[k*NUM_EXAMPLES, :]
        h_p = h[k*NUM_EXAMPLES + 1, :]
        h_Q = h[k*NUM_EXAMPLES + 2 : (k+1)*NUM_EXAMPLES, :]

        loss = get_multimargin_loss(h_q, h_p, h_Q)
        avg_loss += loss.data[0]
        loss.backward(retain_graph=True)
      avg_loss /= batch_size
      optimizer.step()
      if j % (250) == 0:
        print "batch number %d, loss %f, %f seconds" % (j, avg_loss, time.time()-start)
    eval_model(model, data, model_type, False)
    eval_model(model, data, model_type, True)
    print "Epoch ended after %f seconds" % (time.time()-start)

def eval_model(model, data, model_type, use_dev, use_title=True, use_body=True):
  print "Evaluating %s on %s dataset..." % (model_type.name, 'dev' if use_dev else 'test')
  ranked_scores = []
  num_batches = len(data.dev_data) if use_dev else len(data.test_data)
  for i in xrange(num_batches):
    title, body, similar = data.get_next_eval_feature(use_dev)
    h = run_model(model, title, body, use_title, use_body, model_type)
    candidate_scores = []
    # The candidates are all results after the first one, which is h_q.
    h_q = h[0]
    for c in h[1:]:
      candidate_scores.append(get_cosine_similarity(h_q, c))
    # Sort candidate scores in decreasing order and remember which are the
    # correct similar questions.
    ranked_index = np.array(candidate_scores).argsort()[::-1]
    ranked_score = np.isin(ranked_index, similar).astype(int)
    ranked_scores.append(ranked_score)
  eval_obj = Eval(np.array(ranked_scores))
  print "MAP:", eval_obj.MAP()
  print "MRR:", eval_obj.MRR()
  print "Precision@1:", eval_obj.Precision(1)
  print "Precision@5:", eval_obj.Precision(5)

def part1(askubuntu_data, model_type, android_data=None):
  """
  Runs the model from part 1.

  If android_data is not None, also evaluates the model on the android_data
  for the direct transfer section of part 2.
  """
  model = None
  if model_type == ModelType.LSTM:
    model = LSTMEncoder(EMBEDDING_LENGTH, LSTM_HIDDEN_DIM,
                       use_cuda=USE_CUDA, return_average=True)
  elif model_type == ModelType.CNN:
    model = CNNEncoder(EMBEDDING_LENGTH, CNN_HIDDEN_DIM, FILTER_WIDTH,
                     use_cuda=USE_CUDA, return_average=True)
  else:
    print "Error: unknown model type", model_type
    return
  train_model(model_type, askubuntu_data, model, NUM_EPOCHS, BATCH_SIZE,
                     use_title=True, use_body=True)
  # Add in the evaluation on android for direct transfer baseline for part 2.
  if android_data is not None:
    print "----------Evaluating Part1 model on android dataset..."
    eval_part2(model, android_data, True, model_type, using_part1_model=True)
    eval_part2(model, android_data, False, model_type, using_part1_model=True)

def part2(askubuntu_data, android_data, num_epochs, batch_size,
          model_type=ModelType.CNN):
  assert batch_size % 2 == 0
  torch.manual_seed(1)
  model = AdversarialDomainAdaptation(EMBEDDING_LENGTH, CNN_HIDDEN_DIM,
                                      FILTER_WIDTH, LSTM_HIDDEN_DIM,
                                      LAMBDA, USE_CUDA)
  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
  num_training_examples = min(len(askubuntu_data.training_examples),
                              len(android_data.corpus.keys()))
  for i in xrange(num_epochs):
    num_batches = num_training_examples / batch_size
    model.change_lambda(LAMBDA*(i+1))
    print 'new lambda:', LAMBDA*(i+1)
    print "Epoch number %d of %d with %d batches" % (i + 1, num_epochs, num_batches)
    for j in xrange(num_batches):
      optimizer.zero_grad()

      # Get batch_size/2 of askubuntu_data, and batch_size/2 of android_data.
      askubuntu_title, askubuntu_body = \
        askubuntu_data.get_next_training_feature(batch_size/2, True, True)
      android_title, android_body = \
        android_data.get_next_training_feature(batch_size/2 * NUM_EXAMPLES,
                                               True, True)
      # Might need to re-pad so askubuntu and android data have same length.
      if askubuntu_title[0].shape[1] > android_title[0].shape[1]:
        android_title = askubuntu_data.pad_helper(
          android_title[0], android_title[1], batch_size/2 * NUM_EXAMPLES,
          askubuntu_title[0].shape[1]
        )
      elif android_title[0].shape[1] > askubuntu_title[0].shape[1]:
        askubuntu_title = askubuntu_data.pad_helper(askubuntu_title[0],
          askubuntu_title[1], batch_size/2 * NUM_EXAMPLES,
          android_title[0].shape[1]
        )
      if askubuntu_body[0].shape[1] > android_body[0].shape[1]:
        android_body = askubuntu_data.pad_helper(android_body[0],
          android_body[1], batch_size/2 * NUM_EXAMPLES,
          askubuntu_body[0].shape[1]
        )
      elif android_body[0].shape[1] > askubuntu_body[0].shape[1]:
        askubuntu_body = askubuntu_data.pad_helper(askubuntu_body[0],
          askubuntu_body[1], batch_size/2 * NUM_EXAMPLES,
          android_body[0].shape[1]
        )

      # Sanity check.
      # assert askubuntu_title[0].shape == android_title[0].shape
      # assert askubuntu_title[1].shape == android_title[1].shape
      # assert askubuntu_body[0].shape == android_body[0].shape
      # assert askubuntu_body[1].shape == askubuntu_body[1].shape

      # Randomly select an order to mix the askubuntu and android data.
      askubuntu_indexes = random.sample(xrange(batch_size), batch_size/2)
      real_domain_labels = np.zeros(batch_size * NUM_EXAMPLES)
      for k in askubuntu_indexes:
        real_domain_labels[k*NUM_EXAMPLES:(k+1)*NUM_EXAMPLES] = np.ones(NUM_EXAMPLES)
      # Create empty containers to store the mixed data.
      title_vectors = np.ndarray((
        batch_size*NUM_EXAMPLES, askubuntu_title[0].shape[1], EMBEDDING_LENGTH))
      title_masks = np.ndarray((
        batch_size*NUM_EXAMPLES, askubuntu_title[0].shape[1]))
      body_vectors = np.ndarray((
        batch_size*NUM_EXAMPLES, askubuntu_body[0].shape[1], EMBEDDING_LENGTH))
      body_masks = np.ndarray((
        batch_size*NUM_EXAMPLES, askubuntu_body[0].shape[1]))

      # Mix askubuntu and android data into the containers.
      askubuntu_idx = 0
      android_idx = 0
      for k in xrange(batch_size):
        k_start = k * NUM_EXAMPLES
        k_end = (k + 1) * NUM_EXAMPLES

        if real_domain_labels[k*NUM_EXAMPLES] == 1:
          start = askubuntu_idx
          end = askubuntu_idx + NUM_EXAMPLES
          title_vectors[k_start:k_end] = askubuntu_title[0][start:end]
          title_masks[k_start:k_end] = askubuntu_title[1][start:end]
          body_vectors[k_start:k_end] = askubuntu_body[0][start:end]
          body_masks[k_start:k_end] = askubuntu_body[1][start:end]
          askubuntu_idx += NUM_EXAMPLES
        else:
          start = android_idx
          end = android_idx + NUM_EXAMPLES
          title_vectors[k_start:k_end] = android_title[0][start:end]
          title_masks[k_start:k_end] = android_title[1][start:end]
          body_vectors[k_start:k_end] = android_body[0][start:end]
          body_masks[k_start:k_end] = android_body[1][start:end]
          android_idx += NUM_EXAMPLES

      # Run data through the model.
      embeddings, predicted_domain_labels = run_part2_model(model,
        title_vectors, body_vectors, title_masks, body_masks, model_type)

      # Get multimargin loss.
      avg_mm_loss = 0
      for k in xrange(batch_size):
        if real_domain_labels[k * NUM_EXAMPLES] == 1:
          h_q = embeddings[k * NUM_EXAMPLES, :]
          h_p = embeddings[k * NUM_EXAMPLES + 1, :]
          h_Q = embeddings[k * NUM_EXAMPLES + 2 : (k + 1) * NUM_EXAMPLES, :]
          multimargin_loss = get_multimargin_loss(h_q, h_p, h_Q)
          avg_mm_loss += multimargin_loss
          multimargin_loss.backward(retain_graph=True)
      avg_mm_loss /= (batch_size/2) # Only half of the data counts.

      # Use BCELoss for the domain classification.
      domain_loss_func = torch.nn.BCEWithLogitsLoss()
      real_domain_labels = Variable(torch.Tensor(real_domain_labels)
        .type(DOUBLE_DTYPE))
      domain_loss = domain_loss_func(predicted_domain_labels.squeeze(1),
                                     real_domain_labels)
      domain_loss.backward()
      optimizer.step()

      if j % 250 == 0:
        print 'batch %d' % j
        print 'multimargin loss %f, domain loss %f' % (avg_mm_loss.cpu().data.numpy()[0], domain_loss.cpu().data.numpy()[0])

    # At the end of each epoch, evaluate.
    eval_part2(model, android_data, True, model_type)
    eval_part2(model, android_data, False, model_type)

def part3(askubuntu_data, model_type, android_data):
  """
  Runs the model from part 3.

  If android_data is not None, also evaluates the model on the android_data
  for the direct transfer section of part 2.
  """
  model = None
  if model_type == ModelType.LSTM:
    model = LSTMEncoder(EMBEDDING_LENGTH, LSTM_HIDDEN_DIM,
                       use_cuda=USE_CUDA, return_average=True)
  elif model_type == ModelType.CNN:
    model = CNNEncoder(EMBEDDING_LENGTH, CNN_HIDDEN_DIM, FILTER_WIDTH,
                     use_cuda=USE_CUDA, return_average=True)
  else:
    print "Error: unknown model type", model_type
    return
  train_model(model_type, askubuntu_data, model, NUM_EPOCHS, BATCH_SIZE,
              use_title=True, use_body=True, tfidf_weighting=True)
  print "----------Evaluating Part 2.3 on android dataset..."
  eval_part2(model, android_data, True, model_type, using_part1_model=True,
             tfidf_weighting=True)
  eval_part2(model, android_data, False, model_type, using_part1_model=True,
             tfidf_weighting=True)

def run_part2_model(model, title_vectors, body_vectors, title_masks,
                     body_masks, model_type, use_domain_classifier=True):
  """
  Return the average title/body embeddings and the predict domain
  classification labels.
  """
  # If using cnn, must swap axes to get input into the correct shape.
  if model_type == ModelType.CNN:
    title_vectors = np.swapaxes(title_vectors, 1, 2)
    body_vectors = np.swapaxes(body_vectors, 1, 2)
  # Turn things into variables and run through the model.
  title_vectors_var = Variable(torch.Tensor(title_vectors).type(FLOAT_DTYPE))
  title_masks_var = Variable(torch.Tensor(title_masks).type(FLOAT_DTYPE))
  body_vectors_var = Variable(torch.Tensor(body_vectors).type(FLOAT_DTYPE))
  body_masks_var = Variable(torch.Tensor(body_masks).type(FLOAT_DTYPE))
  embeddings, predicted_domain_labels = model.forward(title_vectors_var,
    body_vectors_var, title_masks_var, body_masks_var,
    model_type==ModelType.CNN, use_domain_classifier)
  return embeddings, predicted_domain_labels

def eval_part2(model, android_data, use_dev, model_type,
               using_part1_model=False, batch_size=1, tfidf_weighting=False):
  print "Begin eval_part2..."
  auc_eval = AUCMeter()
  num_batches = len(android_data.dev_data) / batch_size if use_dev \
                else len(android_data.test_data) / batch_size
  for i in xrange(num_batches):
    title, body, similar = android_data.get_next_eval_feature(use_dev,
      tfidf_weighting=tfidf_weighting)
    h = None
    if using_part1_model:
      h = run_model(model, title, body, True, True, model_type)
    else:
      title_vectors, title_masks = title
      body_vectors, body_masks = body
      h, _ = run_part2_model(model, title_vectors, body_vectors, title_masks,
                          body_masks, model_type, False)
    candidate_scores = []
    # The candidates are all results after the first one, which is h_q.
    h_q = h[0]
    for c in h[1:]:
      candidate_scores.append(get_cosine_similarity(h_q, c))
    # Get the correct labels.
    # (1 if the candidate is similar to query question, 0 otherwise.)
    labels = np.zeros(len(candidate_scores))
    for similar_idx in similar:
      labels[similar_idx] = 1
    auc_eval.add(np.array(candidate_scores), labels)
  print "Part 2 AUC for %s: %f" %("dev" if use_dev else "test", auc_eval.value(.05))


def unsupervised_methods(android_data):
  """
  Evaluate Tf-Idf for both dev and test sets.
  """
  def unsupervised_methods_helper(android_data, use_dev):
    auc_eval = AUCMeter()
    batch_size = 1
    num_batches = len(android_data.dev_data) / batch_size if use_dev \
                  else len(android_data.test_data) / batch_size
    for i in xrange(num_batches):
      bows, labels = android_data.get_next_eval_bow_feature(use_dev, batch_size)
      for j in xrange(batch_size):
        # TODO: this currently only works when batch size is 1, fix indexing
        query = bows[0]
        scores = []
        for sample in bows[1:]:
          scores.append(get_tfidf_cosine_similarity(query, sample))
        assert len(scores) == len(labels[j])
        auc_eval.add(np.array(scores), labels[j])
    # Report AUC.
    print "AUC for %s: %f" %("dev" if use_dev else "test", auc_eval.value(.05))

  unsupervised_methods_helper(android_data, True)
  unsupervised_methods_helper(android_data, False)




if __name__ == "__main__":
  if USE_CUDA:
    print "using CUDA"

  """
  Load all the data!
  """
  askubuntu_data = AskUbuntuDataset()
  askubuntu_data.load_corpus("../data/askubuntu/text_tokenized.txt")
  askubuntu_data.init_tfidf_bow_vectors()
  askubuntu_data.load_vector_embeddings("../data/glove/glove_pruned_300D.txt")
  askubuntu_data.load_training_examples("../data/askubuntu/train_random.txt")
  askubuntu_data.load_dev_data("../data/askubuntu/dev.txt")
  askubuntu_data.load_test_data("../data/askubuntu/test.txt")

  android_data = AndroidDataset()
  android_data.load_corpus("../data/android/corpus.tsv")
  android_data.init_tfidf_bow_vectors()
  android_data.load_vector_embeddings("../data/glove/glove_pruned_300D.txt")

  android_data.load_dev_data("../data/android/dev.pos.txt", "../data/android/dev.neg.txt")
  android_data.load_test_data("../data/android/test.pos.txt", "../data/android/test.neg.txt")

  """
  Run all the models!
  """
  # Run part 1 and get direct transfer.
  part1(askubuntu_data, ModelType.CNN, android_data)

  # Get unsupervised benchmark for part 2.
  unsupervised_methods(android_data)

  # Run part 2, the adversarial domain adaptation network.
  # NOTE batch_size must be an even number here!
  part2(askubuntu_data, android_data, num_epochs=20, batch_size=16,
        model_type=ModelType.LSTM)

  # Run our new idea for domain adaptation, which involves tfidf weighting.
  part3(askubuntu_data, ModelType.CNN, android_data)

  print "\n\nDone."


