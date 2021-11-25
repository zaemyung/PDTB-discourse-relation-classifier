import pickle
import random
from itertools import combinations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          RobertaForSequenceClassification,
                          XLMRobertaForSequenceClassification)


class PDTBAllRelationDataModule(pl.LightningDataModule):

    def __init__(
        self,
        model_name_or_path: str = 'roberta-large-mnli',
        data_pickle_paths: list = ['./data/relations_and_documents_en.pkl'],
        valid_size: int = 2000,
        test_size: int = 2000,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        random_seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.data_pickle_paths = data_pickle_paths
        self.valid_size = valid_size
        self.test_size = test_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        print(self.tokenizer.model_max_length)

        self.first_classes = {
            'NoClass': 0,
            'Temporal': 1,
            'Causation': 2,
            'Conditional': 3,
            'Purpose': 4,
            'Contrast': 5,
            'Conjunction': 6,
            'Disjunction': 7,
            'Expansion': 8,
            'Concession': 9,  # only for PDTB
            'Similarity': 10,  # only for PDTB
        }

    def extract_relations(self, doc_data):
        unfil_relations, sentences = doc_data
        rel_sent_ind_pairs = set()
        relations = []
        for rel in unfil_relations:
            ind_pair_1 = (rel['arg1_sentence_index'], rel['arg2_sentence_index'])
            ind_pair_2 = (rel['arg2_sentence_index'], rel['arg1_sentence_index'])
            rel_sent_ind_pairs.add(ind_pair_1)
            rel_sent_ind_pairs.add(ind_pair_2)
            if 'sclass1a' not in rel or rel['label'] is None:
                continue
            relations.append(rel)
        all_ind_pairs = set(combinations(range(len(sentences)), 2))
        # do not pick sentence pairs where there is a relation
        noclass_pairs = all_ind_pairs - rel_sent_ind_pairs
        # do not pick consecutive sentences
        noclass_pairs = [pair for pair in noclass_pairs if abs(pair[0] - pair[1]) > 1]
        max_num_noclass_pairs = int(len(relations) / 6)
        noclass_relations = []
        for i, j in noclass_pairs:
            if len(noclass_relations) >= max_num_noclass_pairs:
                break
            arg1_ind = i if i <= j else j
            arg2_ind = j if i <= j else i
            arg1_sent = sentences[arg1_ind]
            arg2_sent = sentences[arg2_ind]
            # do not pick short sentences
            if len(arg1_sent.split()) < 3 or len(arg2_sent.split()) < 3:
                continue
            rel = {
                'arg1_sentence_index': arg1_ind,
                'arg2_sentence_index': arg2_ind,
                'arg1_sentence': arg1_sent,
                'arg2_sentence': arg2_sent,
                'sclass1a': 'NoClass',
                'label': 'NoClass'
            }
            noclass_relations.append(rel)

        relations.extend(noclass_relations)
        random.shuffle(relations)
        return relations

    def setup(self, stage = None):
        if stage == 'train' or stage is None:
            dataset = []
            for d_p in self.data_pickle_paths:
                with open(d_p, 'rb') as f:
                    dataset_ = pickle.load(f)
                    assert isinstance(dataset_, list)
                    dataset.extend(dataset_)

            processed = []
            for data in tqdm(dataset):
                relations = self.extract_relations(data)
                relations = [self.convert_to_features(rel) for rel in relations]
                processed.extend(relations)

            # split dataset
            random.shuffle(processed)
            self.dataset = {
                'validation': processed[:self.valid_size],
                'test': processed[self.valid_size:self.valid_size + self.test_size],
                'train': processed[self.valid_size + self.test_size:],
            }
            print(f'len(self.dataset["train"]): {len(self.dataset["train"])}')
            print(f'len(self.dataset["validation"]): {len(self.dataset["validation"])}')
            print(f'len(self.dataset["test"]): {len(self.dataset["test"])}')

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, collate_fn=self.collate)

    def convert_to_features(self, data):
        # e.g. [0, 31414, 6, 127, 766, 16, 2084, 139, 2, 2, 34033, 7, 972, 47, 2]
        encoded = self.tokenizer(data['arg1_sentence'], data['arg2_sentence'], truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt')['input_ids'][0]
        # e.g. Comparison
        label = data['label']
        label_id = self.first_classes[label]
        return encoded, label_id

    def collate(self, list_of_features):
        all_encoded, all_label = zip(*list_of_features)
        all_encoded_len = torch.tensor([len(x) for x in all_encoded], dtype=torch.long)
        padded_encoded = pad_sequence(all_encoded, batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        maxlen = padded_encoded.size(1)
        attention_mask = (torch.arange(maxlen)[None, :] < all_encoded_len[:, None]).long()
        all_label = torch.tensor(all_label, dtype=torch.long)
        return padded_encoded, attention_mask, all_label


class PairwiseDiscourseModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        model_arch: str,
        learning_rate=1e-3,
        adam_epsilon=1e-7,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name_or_path)
        if model_arch == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)
        elif model_arch == 'xlm_roberta':
            self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f'Unknown model arch: {model_arch}')
        config.num_labels = num_labels
        self.model.num_labels = num_labels
        self.model.classifier = ClassificationHead(config)

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            x = {
                'input_ids': inputs[0],
                'attention_mask': inputs[1]
            }
            if len(inputs) == 3:
                x['labels'] = inputs[2]

        return self.model(**x)

    def training_step(self, batch, batch_idx):
        x = {
            'input_ids': batch[0],
            'attention_mask': batch[1]
        }
        if len(batch) == 3:
            x['labels'] = batch[2]
        outputs = self.model(**x)
        loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = x['labels']
        acc = torchmetrics.functional.accuracy(preds, labels)
        metrics = {'acc': acc, 'loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.training_step(batch, batch_idx)
        metrics = {'val_loss': metrics['loss'], 'val_acc': metrics['acc']}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_loss': metrics['val_loss'], 'test_acc': metrics['val_acc']}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        try:
            dropout = config.hidden_dropout_prob
        except AttributeError:
            dropout = config.dropout
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def train_roberta_baseline_en():
    dm = PDTBAllRelationDataModule(data_pickle_paths=['./data/relations_and_documents_en.pkl'])
    dm.prepare_data()
    dm.setup()
    model = PairwiseDiscourseModel(model_name_or_path='roberta-large-mnli', num_labels=11, model_arch='roberta')
    checkpoint_callback = ModelCheckpoint(monitor='val_acc')
    early_stopping = EarlyStopping('val_loss', patience=6)
    # trainer = pl.Trainer(gpus=4, accelerator='ddp_spawn', max_epochs=100, callbacks=[checkpoint_callback, early_stopping])
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback, early_stopping], auto_lr_find=True)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    train_roberta_baseline_en()
