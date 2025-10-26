import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerFast, T5TokenizerFast

import numpy as np

# this is old and had an import error so just copied the source function
#from transformers.tokenization_utils import trim_batch
def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def trim_protoroles(
    protoroles,
    pad_token_id
):
    """Remove columns that are populated exclusively by pad_token_id"""
    pad_vector = torch.full((1,1,14), pad_token_id)
    keep_column_mask = protoroles.ne(pad_vector).any(dim=0)
    protoroles_new = []
    for i in range(protoroles.shape[1]):
        mask = keep_column_mask[i,:]
        assert sum(mask) == 14 or sum(mask) == 0
        if sum(mask) == 14:
            sliced = protoroles[:, i, mask].tolist()
            protoroles_new.append(sliced)
    protoroles = torch.tensor(protoroles_new).permute(1, 0, 2)
    return protoroles

def encode_dataset(tokenizer, dataset, max_length=520, padding='max_length', return_tensors="pt"):
    tokenized = tokenizer(dataset['text'], max_length=max_length, 
        padding=padding, truncation=True, return_tensors=return_tensors, return_offsets_mapping=True)

    tokenized['text'] = dataset['text']
    tokenized['text_id'] = dataset['text_id']

    labels = tokenized.input_ids.detach().clone()
    tokenized['labels'] = labels

    # arg_ind: [len_dataset, char_seq_size] BxL
    # protoroles: [len_dataset, seq_size, 14] BxLx*
    arg_ind = []
    protoroles = []
    for sent_idx in range(len(tokenized['offset_mapping'])):
        sent_arg_ind_bpe = []
        sent_protoroles_bpe = []
        sent_offset_mapping = tokenized['offset_mapping'][sent_idx]
        sent_arg_ind_char = dataset['arg_ind'][sent_idx]
        sent_protoroles_char = dataset['protoroles_char'][sent_idx]
        
        for token_idx in sent_offset_mapping:
            start = token_idx[0]
            stop = token_idx[1]
            if start != stop:
                arg_ind_list = sent_arg_ind_char[start:stop]
                assert all(ind == arg_ind_list[0] for ind in arg_ind_list)
                sent_arg_ind_bpe.append(arg_ind_list[0])

                protoroles_list = sent_protoroles_char[start:stop]
                ref_vec = protoroles_list[0]
                for pr_vec in protoroles_list:
                    for i in range(14):
                        assert pr_vec[i] == ref_vec[i]
                sent_protoroles_bpe.append(protoroles_list[0])

        # for eos token
        sent_arg_ind_bpe.append(0)
        arg_ind.append(torch.tensor(sent_arg_ind_bpe))
        empty_protoroles_vector = torch.zeros(14)
        sent_protoroles_bpe.append(empty_protoroles_vector)
        sent_protoroles_bpe = [tensor.tolist() for tensor in sent_protoroles_bpe]
        protoroles.append(torch.tensor(sent_protoroles_bpe))

    # B, T
    tokenized['arg_ind'] = nn.utils.rnn.pad_sequence(arg_ind, batch_first=True, padding_value=2)[:, :max_length]
    # B, T, 14
    tokenized['protoroles'] = nn.utils.rnn.pad_sequence(protoroles, batch_first=True, padding_value=2)[:, :max_length, :]

    list_dataset = []
    for h in range(len(tokenized['input_ids'])):
        input_ids = tokenized['input_ids'][h]
        attention_mask = tokenized['attention_mask'][h]
        labels = tokenized['labels'][h]
        arg_ind = tokenized['arg_ind'][h]
        protoroles = tokenized['protoroles'][h]

        dataset_item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "arg_ind": arg_ind, "protoroles": protoroles}
        list_dataset.append(dataset_item)

    return list_dataset

class UDSDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_file
    ):
        super().__init__()
        self.tokenizer = tokenizer
        #self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.max_length = 520

        with open(data_file) as j:
            uds = json.load(j)
            uds_examples = uds['data']
            uds_sentences = {'text':[], 'text_id':[], 'protoroles_char':[], 'arg_ind':[]}
            for uds_example in uds_examples:
                sent = uds_example['sentence']
                uds_sentences['text'].append(sent)

                sent_id = uds_example['sent_id']
                uds_sentences['text_id'].append(sent_id)

                # [char_size, 14]
                protoroles_char = torch.tensor(uds_example['protoroles_char'])
                # [len_dataset, char_size, 14]
                uds_sentences['protoroles_char'].append(protoroles_char)

                # [seq_size]
                arg_ind = torch.tensor(uds_example['arg_ind'])
                # [len_dataset, seq_size]
                uds_sentences['arg_ind'].append(arg_ind)

        self.dataset = encode_dataset(self.tokenizer, uds_sentences, max_length=self.max_length)

    def __iter__(self):
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        dp = self.dataset[index]
        input_ids = dp['input_ids']
        attention_mask = dp['attention_mask']
        labels = dp['labels']
        arg_ind = dp['arg_ind']
        protoroles = dp['protoroles']

        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        labels = labels[:self.max_length]
        arg_ind = arg_ind[:self.max_length]
        protoroles = protoroles[:self.max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "arg_ind": arg_ind, "protoroles": protoroles}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id, trim_y=True):
        if trim_y:
            y = trim_batch(batch["labels"], pad_token_id)
        else:
            y = batch["labels"]
        input_ids, attention_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return input_ids, attention_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_masks = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        arg_inds = torch.stack([x["arg_ind"] for x in batch])
        protoroles = torch.stack([x["protoroles"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(labels, pad_token_id)
        arg_inds = trim_batch(arg_inds, 2)
        input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_masks)
        protoroles = trim_protoroles(protoroles, 2)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": y, "arg_ind": arg_inds, "protoroles": protoroles}