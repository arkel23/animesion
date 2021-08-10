import ast
import pickle
import argparse
import pandas as pd
from collections import Counter


class CustomTokenizer(object):
    def __init__(self, vocab, max_text_seq_len, shuffle):
        self.max_text_seq_len = max_text_seq_len
        self.vocab = vocab
        self.shuffle = shuffle

    def __call__(self, tag_list):
        no_tokens = len(tag_list) + 2
        diff = abs(self.max_text_seq_len - no_tokens)

        tokens = []
        tokens.append(self.vocab('[CLS]'))

        if no_tokens > self.max_text_seq_len:
            tokens.extend([self.vocab(tag) for tag in tag_list[:self.max_text_seq_len-2])
            tokens.append(self.vocab('[SEP]'))
        elif no_tokens < self.max_text_seq_len:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))
            tokens.extend([self.vocab('[PAD]') for _ in range(diff)])
        else:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))

        return torch.tensor(tokens, dtype=torch.int64)


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(args):
    """Build a simple vocabulary wrapper."""
    df = pd.read_csv(args.caption_path, usecols=['tags_cat0'])
    counter = Counter()
    for i, tag in enumerate(df.tags_cat0):
        tokens = ast.literal_eval(tag)
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    print('Number of unique tokens: ', len(counter))
    print('Total number of tokens: ', sum(counter.values()))
    words = [word for word, cnt in counter.items() if cnt >= args.threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('[PAD]')
    vocab.add_word('[CLS]')
    vocab.add_word('[SEP]')
    vocab.add_word('[UNK]')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print("Total vocabulary size: {}".format(len(vocab)))

    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, required=True,
        help='Path for dataframe file with whole list of files and tags')
    parser.add_argument('--vocab_path', type=str, default='./utilities/vocab.pkl')
    parser.add_argument('--threshold', type=int, default=2, 
        help='Minimum tag count to put into final vocabulary')
    args = parser.parse_args()

    vocab = build_vocab(args)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Saved vocabulary wrapper to ', args.vocab_path)

if __name__ == '__main__':
    main()

