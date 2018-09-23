import tensorflow as tf
import os
import argparse
from model.input_fn import input_fn
from model.model_fn import model_fn
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-s', '--sample', action='store_true')

parser.set_defaults(sample=False)


params = {'rnn_size': 256,
          'keep_prob': 0.8,
          'embedding_size': 300,
          'num_layers': 1,
          'save_summary_steps': 50,
          'vocabulary_path': 'data/words.txt',
          'num_epochs': 1,
          'batch_size': 8,
          'learning_rate': 0.001}


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parser.parse_args()

    with open(params['vocabulary_path'], 'r') as file:
        num_lines = 0
        for _ in file:
            num_lines += 1

    params['vocab_size'] = num_lines + 1
    params['end_id'] = num_lines - 1
    params['go_id'] = num_lines - 2

    config = tf.estimator.RunConfig(tf_random_seed=43,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params['save_summary_steps'])

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    data_path = os.path.join(args.data_dir, 'train.csv')

    if args.sample:
        nrows = 100
        params['train_size'] = nrows
    else:
        nrows = None
        params['train_size'] = int(pd.read_csv(data_path).shape[0] * 0.9)

    estimator.train(lambda: input_fn(data_path, params, is_training=True, nrows=nrows))
