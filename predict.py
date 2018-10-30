import tensorflow as tf
import os
import argparse
from model.input_fn import input_fn
from model.model_fn import model_fn
from tqdm import tqdm
from train import params

parser = argparse.ArgumentParser()
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-s', '--sample', action='store_true')

parser.set_defaults(sample=False)

# TODO: импортить словарь из train.py, менять keep_prob
params = {'rnn_size': 256,
          'keep_prob': 1,
          'embedding_size': 300,
          'num_layers': 1,
          'save_summary_steps': 50,
          'vocabulary_path': 'data/words.txt',
          'num_epochs': 1,
          'batch_size': 8,
          'learning_rate': 0.001}


def decode_preds(probs, vocab):
    words = []
    for row in probs:
        word = vocab[row]
        words.append(word)

    return ' '.join(words)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parser.parse_args()

    vocab = []
    with open(params['vocabulary_path'], 'r') as file:
        for line in file:
            vocab.append(line.replace('\n', ''))

    num_lines = len(vocab)
    params['vocab_size'] = num_lines + 1
    params['end_id'] = num_lines - 1
    params['go_id'] = num_lines - 2

    config = tf.estimator.RunConfig(tf_random_seed=43,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params['save_summary_steps'])

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    if args.sample:
        dp = os.path.join(args.data_dir, 'sample', 'eval.csv')
        if os.path.exists(dp):
            data_path = dp
        else:
            data_path = os.path.join(args.data_dir, 'eval.csv')
    else:
        data_path = os.path.join(args.data_dir, 'eval.csv')

    tf.logging.info("Predicting the data...")
    train_predictions = estimator.predict(lambda: input_fn(data_path, params, is_training=False))

    preds = []
    for i, p in tqdm(enumerate(train_predictions)):
        if args.sample and i > 100:
            break
        probs = p['preds']
        gen_line = decode_preds(probs, vocab)
        preds.append(gen_line)

    print('Predictions: \n\n{}'.format('\n'.join(preds)))

    tf.logging.info("Saving the data...")
    with open(os.path.join(args.data_dir, 'results.txt'), 'w') as file:
        for p in preds:
            file.write(f'{p}\n')
