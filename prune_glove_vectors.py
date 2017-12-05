"""
A script to prune the off the shelf glove vectors to only include those
that are in the Askubuntu or Android datasets.
"""

out_filename = "data/glove/glove_pruned_200D.txt"
missing_words_filename = "data/glove/glove_pruned_missing_words.txt"
glove_input_filename = "data/glove/glove.6B.200D.txt"
askubuntu_filename = "data/askubuntu/text_tokenized.txt"
android_filename = "data/android/corpus.tsv"

if __name__ == "__main__":
  # Combine askubuntu and android corpora into one set.
  joint_corpus = {}

  def add_to_joint_corpus(corpus_filename):
    corpus_file = open(corpus_filename, "r")
    for line in corpus_file:
      _, title, body = line.split("\t")
      for word in title.split():
        joint_corpus[word] = True
      for word in body.split():
        joint_corpus[word] = True
    corpus_file.close()

  add_to_joint_corpus(askubuntu_filename)
  add_to_joint_corpus(android_filename)
  print "Number of words in joint corpus: %d" % len(joint_corpus.keys())

  # Go through glove vectors, copy those that are in joint corpus
  # to the outfile.
  glove_input_file = open(glove_input_filename, "r")
  output_file = open(out_filename, "w")
  added_words_count = 0
  added_words = {}
  for line in glove_input_file:
    word = line.split()[0]
    if word in joint_corpus or word == "unk":
      if word == "unk":
        print '"unk" added'
      output_file.write(line)
      added_words_count += 1
      added_words[word] = True
  glove_input_file.close()
  output_file.close()
  print "Added %d words to output file at %s " % (added_words_count, out_filename)

  # Also record which words are missing.
  missing_words_file = open(missing_words_filename, "w")
  for word in joint_corpus.keys():
    if word not in added_words:
      missing_words_file.write(word+"\n")
  missing_words_file.close()
  print "%d missing words written to %s" % (len(joint_corpus.keys()) - added_words_count, missing_words_filename)
  print "Done."
