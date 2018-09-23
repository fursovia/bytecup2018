import json
import pandas as pd
from multiprocessing import Pool
import re
from collections import Counter
import argparse
import os
from glob import glob
import gc
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')


def get_data(line):
    dict_ = json.loads(line[:-2])  # там в конце \n
    return dict_['id'], dict_['content'], dict_['title']


def tokenize_string(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip()


def update_vocabulary(tokens, counter):
    counter.update(tokens)


def create_vocab(pd_series, min_count_word=5):
    vocabulary = Counter()
    _ = pd_series.apply(lambda x: update_vocabulary(x.casefold().split(), vocabulary))
    vocabulary = [tok for tok, count in vocabulary.items() if count >= min_count_word]
    vocabulary += ['<PAD>']
    vocabulary += ['<GO>']
    vocabulary += ['<EOS>']
    return vocabulary


def save_vocab_to_txt_file(vocab, txt_path):
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


if __name__ == '__main__':

    args = parser.parse_args()

    print('Reading the data')
    filenames = glob(os.join(args.data_dir, '*[0-9].txt'))

    data = []
    for file_name in filenames:
        with open(file_name, 'r') as file:
            d = file.readlines()
            data.extend(d)

    with Pool(10) as p:
        new_data = p.map(get_data, data)

    del data
    gc.collect()

    data = pd.DataFrame(new_data, columns=['id', 'content', 'title'])
    num_ex = data.shape[0]

    data['content'] = data['content'].apply(lambda string: tokenize_string(string).casefold())
    data['title'] = data['title'].apply(lambda string: tokenize_string(string).casefold())

    print('Dropping blank examples')
    data.drop(index=data[(data['content'].apply(lambda s: len(s.split())) == 0) |
                         (data['title'].apply(lambda s: len(s.split())) == 0)].index, inplace=True)

    data['title'] = '<GO> ' + data['title'] + ' <EOS>'

    print('{} examples dropped'.format(num_ex - data.shape[0]))

    data_tr, data_ev = train_test_split(data, test_size=0.1, random_state=24)
    train_path = os.path.join('data', 'train.csv')
    eval_path = os.path.join('data', 'eval.csv')

    data_tr.to_csv(train_path, index=False)
    data_ev.to_csv(eval_path, index=False)

    print('Creating vocab...')
    vocabulary = create_vocab(pd.concat((data['content'], data['title'])))
    save_vocab_to_txt_file(vocabulary, 'data/words.txt')
