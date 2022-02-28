import os
import pickle
import sys
from glob import glob
from itertools import islice

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import RobertaTokenizer

from discourse_baseline import PairwiseDiscourseModel

sys.path.append('..')


def read_lines(file_path):
    lines = []
    with open(file_path, 'r') as inf:
        for line in inf:
            lines.append(line.rstrip('\n'))
    return lines


INDEX_TO_DT = {
    0: '<DT:NoClass>',
    1: '<DT:Temporal>',
    2: '<DT:Causation>',
    3: '<DT:Conditional>',
    4: '<DT:Purpose>',
    5: '<DT:Contrast>',
    6: '<DT:Conjunction>',
    7: '<DT:Disjunction>',
    8: '<DT:Expansion>',
    9: '<DT:Concession>',
    10: '<DT:Similarity>'
}


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

    model_dir = '/tmp-network/user/zaemyung-kim/projects/discourse_style/Discourse-Sentiment/PDTB-discourse-relation-classifier/models'
    model_path = os.path.join(model_dir, 'classifier', 'classifier.ckpt')
    model = PairwiseDiscourseModel.load_from_checkpoint(checkpoint_path=model_path, hparams_file=os.path.join(model_dir, 'classifier', 'hparams.yaml'))
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_dir, 'roberta-large-mnli'), use_fast=True)
    tokenizer.model_max_length = 512
    batch_size = 50

    corpus_dir = '/tmp-network/user/zaemyung-kim/projects/discourse_MT/evaluations/control_discourse_connectives'

    def run_tag(doc_paths):
        print(doc_paths)

        con_to_dt = {}
        with open('con_to_dt.en', 'r') as f:
            for line in f:
                con, dt = line.split('\t')
                con = con.strip()
                dt = dt.strip()
                con_to_dt[con] = dt

        conns = sorted(list(con_to_dt.keys()), key=lambda x:len(x), reverse=True)

        for doc_path in doc_paths:
            doc_lines = read_lines(doc_path)
            sent_pairs = []
            for i in range(len(doc_lines) - 2 + 1):
                sent_pairs.append(doc_lines[i: i + 2])

            print(len(sent_pairs))

            splitted_part = sent_pairs
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
            with open(doc_path[:-2] + 'dt', 'w') as f:
                for data, pred in zip(splitted_part, all_preds):
                    dt = INDEX_TO_DT[pred.item()]
                    if dt == '<DT:NoClass>':
                        for con in conns:
                            second_sent = data[1].lower()
                            if con == second_sent[:len(con)]:
                                dt = con_to_dt[con]
                                break
                    f.write(f'{dt}\n')

    doc_paths = glob(os.path.join(corpus_dir, 'more_implicit_lines', '*.en'))
    run_tag(doc_paths)
    doc_paths = glob(os.path.join(corpus_dir, 'more_explicit_lines', '*.en'))
    run_tag(doc_paths)