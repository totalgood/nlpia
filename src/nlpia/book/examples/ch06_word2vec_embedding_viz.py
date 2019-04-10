# script adopted from https://gist.github.com/lampts/026a4d6400b1efac9a13a3296f16e655

import gensim
import numpy as np
import tensorflow as tf
from nlpia.loaders import get_data
from tensorflow.contrib.tensorboard.plugins import projector

words = ('Sacramento', 'California', 'Oregon', 'Salem', 'Washington', 'Olympia')

# loading your gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

model = get_data('w2v', limit=300000)  # <1>

# project part of vocab, 10K of 300 dimension
w2v_10K = np.zeros((6, 300))
with open("/Users/hannes/Downloads/prefix_metadata.tsv", 'w+') as file_metadata:
    # for i, word in enumerate(model.index2word[:200000]):
    #     w2v_10K[i] = model[word]
    #     file_metadata.write(word.encode('utf-8') + '\n')
    for i, word in enumerate(list(words)):
        w2v_10K[i] = model[word]
        file_metadata.write(word.encode('utf-8') + '\n')

# define the model without training
sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v_10K, trainable=False, name='word2vec_embedding')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter('/Users/hannes/Downloads', sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'word2vec_embedding'
embed.metadata_path = '/Users/hannes/Downloads//prefix_metadata.tsv'

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

saver.save(sess, '/Users/hannes/Downloads/prefix_model.ckpt', global_step=1000)

# open tensorboard with logdir, check localhost:6006 for viewing your embedding.
# tensorboard --logdir="./projector/"
