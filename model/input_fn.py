import tensorflow as tf
import pandas as pd


def build_vocab(file_name):
    words = tf.contrib.lookup.index_table_from_file(file_name,
                                                    num_oov_buckets=1,
                                                    delimiter='\n',
                                                    name='vocab')
    return words


def vectorize(string, vocab):
    splitted = tf.string_split([string]).values
    vectorized = vocab.lookup(splitted)
    return vectorized


def input_fn(data_path, params, is_training=True):
    pd_dframe = tf.contrib.data.CsvDataset(data_path, [tf.int64, tf.string, tf.string], header=True)

    vocab = build_vocab(params['vocabulary_path'])
    id_pad_word = vocab.lookup(tf.constant('<PAD>'))
    fake_padding = tf.constant(20, dtype=tf.int32)

    dataset = pd_dframe.map(lambda i, s, t: (vectorize(s, vocab), vectorize(t, vocab)))
    dataset = dataset.map(lambda s, t: (s, t, tf.shape(s)[-1], tf.shape(t)[-1]))

    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
    padding_values = (id_pad_word, id_pad_word, fake_padding, fake_padding)

    if is_training:
        dataset = dataset.shuffle(params['train_size'])
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.padded_batch(params['batch_size'],
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values,
                                   drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=None)
    dataset = dataset.map(lambda s, t, ss, ts: ({'source': s, 'source_len': ss, 'target_len': ts}, t))

    return dataset
