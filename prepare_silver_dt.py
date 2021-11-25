import os
import pickle
import sys
from itertools import islice

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import RobertaTokenizer

from discourse_baseline import PairwiseDiscourseModel

sys.path.append('..')


def convert_to_features(data):
    # e.g. [0, 31414, 6, 127, 766, 16, 2084, 139, 2, 2, 34033, 7, 972, 47, 2]
    arg1_sentence, arg2_sentence = data
    # print(tokenizer.model_max_length)
    encoded = tokenizer(arg1_sentence, arg2_sentence, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')['input_ids'][0]
    return encoded


def collate(all_encoded):
    all_encoded_len = torch.tensor([len(x) for x in all_encoded], dtype=torch.long)
    padded_encoded = pad_sequence(all_encoded, batch_first=True, padding_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    maxlen = padded_encoded.size(1)
    attention_mask = (torch.arange(maxlen)[None, :] < all_encoded_len[:, None]).long()
    return padded_encoded, attention_mask


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def split_into_n_parts(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('Usage: python prepare_silver_dt.py n_parts split_ind corpus_dir')
    n_parts = int(sys.argv[1])
    split_ind = int(sys.argv[2])
    silver_dir = sys.argv[2]

    batch_size = 50

    version_dir = './models'
    model_path = os.path.join(version_dir, 'roberta-large-mnli_epoch-1_step-5701.ckpt')
    model = PairwiseDiscourseModel.load_from_checkpoint(checkpoint_path=model_path, hparams_file=os.path.join(version_dir, 'hparams.yaml'))
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(version_dir, 'roberta-large-mnli'), use_fast=True)
    tokenizer.model_max_length = 512

    sent_pairs = []
    with open(os.path.join(silver_dir, 'first_sents.en'), 'r') as first_f, open(os.path.join(silver_dir, 'second_sents.en'), 'r') as second_f:
        for first, second in zip(first_f, second_f):
            first = first.strip()
            second = second.strip()
            sent_pairs.append((first, second))

    print(len(sent_pairs))
    print(sent_pairs[:5])

    splitted_parts = list(split_into_n_parts(sent_pairs, n_parts))
    assert len(sent_pairs) == sum([len(c) for c in splitted_parts])

    splitted_part = splitted_parts[split_ind]
    splitted_part_feats = [convert_to_features(data) for data in splitted_part]
    batches = list(chunks(splitted_part_feats, batch_size))
    batches = [collate(batch) for batch in batches]

    all_preds = []
    for batch in tqdm(batches):
        outputs = model(batch)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        all_preds.extend(preds)

    assert len(splitted_part) == len(all_preds)
    for i, (data, pred) in enumerate(zip(splitted_part, all_preds)):
        data = *data, pred.item()
        splitted_part[i] = data

    with open(os.path.join(silver_dir, f'splitted_part_{split_ind}.pkl'), 'wb') as f:
        pickle.dump(splitted_part, f)
