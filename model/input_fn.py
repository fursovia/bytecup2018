import tensorflow as tf
import pandas as pd


def build_vocab(file_name):
    words = tf.contrib.lookup.index_table_from_file(file_name, num_oov_buckets=1, delimiter='\n', name='vocab')
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, words.init)
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
    dataset = load_dataset_from_pd(pd_dframe, vocab)

    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
    padding_values = (id_pad_word, id_pad_word)

    if is_training:
        dataset = dataset.shuffle(params['train_size'])
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.padded_batch(params['batch_size'],
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values,
                                   drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=None)

    return dataset
