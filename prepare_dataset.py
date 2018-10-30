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
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-s', '--sample', action='store_true')
parser.add_argument('-wc', '--word_count', type=int, default=5)

parser.set_defaults(sample=False)


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
    data_dir = args.data_dir
    zip_filenames = glob(os.path.join(data_dir, '*[0-9].zip'))
    txt_filenames = [fn.split(sep='/')[-1][:-4] + '.txt' for fn in zip_filenames]

    data = []
    for i, file_name in enumerate(zip_filenames):
        print('File: {}'.format(file_name))
        print('File inside: {}'.format(txt_filenames[i]))

        with ZipFile(file_name) as myzip:
            with myzip.open(txt_filenames[i]) as myfile:
                d = myfile.readlines()
                data.extend(d)

    print('Reformatting the data...')
    with Pool(10) as p:
        new_data = p.map(get_data, data)

    del data
    gc.collect()

    if args.sample:
        num_els = 10000
        data_dir = os.path.join(data_dir, 'sample')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    else:
        num_els = None

    data = pd.DataFrame(new_data[:num_els], columns=['id', 'content', 'title'])
    num_ex = data.shape[0]

    print('Cleaning the data...')
    data['content'] = data['content'].apply(lambda string: tokenize_string(string).casefold())
    data['title'] = data['title'].apply(lambda string: tokenize_string(string).casefold())

    print('Dropping blank examples')
    data.drop(index=data[(data['content'].apply(lambda s: len(s.split())) == 0) |
                         (data['title'].apply(lambda s: len(s.split())) == 0)].index, inplace=True)

    data['content'] = '<GO> ' + data['content'] + ' <EOS>'
    data['title'] = '<GO> ' + data['title'] + ' <EOS>'

    print('{} examples dropped'.format(num_ex - data.shape[0]))

    data_tr, data_ev = train_test_split(data, test_size=0.1, random_state=24)
    train_path = os.path.join(data_dir, 'train.csv')
    eval_path = os.path.join(data_dir, 'eval.csv')

    data_tr.to_csv(train_path, index=False)
    data_ev.to_csv(eval_path, index=False)

    print('Creating vocab...')
    vocabulary = create_vocab(pd.concat((data['content'], data['title'])), min_count_word=args.word_count)
    save_vocab_to_txt_file(vocabulary, 'data/words.txt')
