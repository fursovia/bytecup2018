import tensorflow as tf
import pandas as pd


def build_vocab(file_name):
    words = tf.contrib.lookup.index_table_from_file(file_name,
                                                    num_oov_buckets=1,
                                                    delimiter='\n',
                                                    name='vocab')
    return words


def preprocess_sentence(sentence, vocab):
    return tf.data.Dataset.from_tensor_slices(sentence)  \
                .map(lambda string: tf.string_split([string]).values)  \
                .map(lambda tokens: (vocab.lookup(tokens)))


def load_dataset_from_pd(pd_dframe, vocab):
    source = preprocess_sentence(pd_dframe['content'].values, vocab)
    target = preprocess_sentence(pd_dframe['title'].values, vocab)
    return tf.data.Dataset.zip((source, target))


def input_fn(data_path, params, is_training=True, nrows=None):

    pd_dframe = pd.read_csv(data_path, nrows=nrows)

    vocab = build_vocab(params['vocabulary_path'])
    id_pad_word = vocab.lookup(tf.constant('<PAD>'))
    fake_padding = tf.constant(20, dtype=tf.int32)

    dataset = load_dataset_from_pd(pd_dframe, vocab)
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
