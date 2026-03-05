import torch
from collections import Counter
from functools import partial
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

MAX_SEQ_LEN = 256
MIN_FREQ = 5


def tokenizer(text):
    return text.lower().split()

class Vocabulary:
    """Maps words to integer indices with special <pad> and <unk> tokens.

    Words appearing fewer than `min_freq` times are treated as <unk>.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq=MIN_FREQ):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

        self._add_word(self.PAD_TOKEN)  # index 0
        self._add_word(self.UNK_TOKEN)  # index 1

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_from_texts(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self._add_word(word)

        print(f"Vocabulary built: {len(self)} words with min_freq={self.min_freq}")

    def encode(self, text):
        tokens = tokenizer(text)
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    @property
    def pad_idx(self):
        return self.word2idx[self.PAD_TOKEN]

    def __len__(self):
        return len(self.word2idx)


class IMDbDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.vocab.encode(self.texts[idx]), self.labels[idx]


def collate_batch(batch, pad_idx):
    text_list, label_list = [], []

    for encoded_text, label in batch:
        if len(encoded_text) > MAX_SEQ_LEN:
            encoded_text = encoded_text[:MAX_SEQ_LEN]
        elif len(encoded_text) < MAX_SEQ_LEN:
            encoded_text = encoded_text + [pad_idx] * (MAX_SEQ_LEN - len(encoded_text))

        text_list.append(encoded_text)
        label_list.append(label)

    return (
        torch.tensor(text_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.float),
    )


def get_dataloaders(batch_size=64):
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    vocab = Vocabulary(min_freq=MIN_FREQ)
    vocab.build_from_texts(train_texts)

    train_dataset = IMDbDataset(train_texts, train_labels, vocab)
    test_dataset = IMDbDataset(test_texts, test_labels, vocab)

    collate_fn = partial(collate_batch, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    return train_loader, test_loader, vocab
