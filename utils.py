import os
import json
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, IterableDataset

from datasets import load_dataset

# this is old and had an import error so just copied the source function
#from transformers.tokenization_utils import trim_batch

def encode_mask_tag(tokenizer, text, max_length=520, padding='max_length', return_tensors="pt", mlm_prob=0.15):
    tokenized = tokenizer(text['text'], max_length=max_length, 
        padding=padding, truncation=True, return_tensors=return_tensors)

    labels = tokenized.input_ids.detach().clone()
    tokenized['labels'] = labels

    rand = torch.rand(tokenized.input_ids.shape)
    mask_arr = (rand < mlm_prob) * (tokenized.input_ids != 1) * (tokenized.input_ids != 0)
    selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
    tokenized.input_ids[0, selection] = 32099 #<extra_id_0>

    return tokenized

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

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(x["input_ids"][0]) for x in batch])
    attention_masks = torch.stack([torch.tensor(x["attention_mask"][0]) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"][0]) for x in batch])
    pad_token_id = 0
    labels = trim_batch(labels, pad_token_id)
    input_ids, attention_masks = trim_batch(input_ids, pad_token_id, attention_mask=attention_masks)
    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


def encode_and_mask_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", max_examples=None):
    examples = []
    if data_path[-3:] == '.gz':
        print('Data file is gzipped')
        f = gzip.open(data_path, "rt")
    else:
        print('Data file is plain text')
        print(data_path)
        f = open(data_path, "r", encoding='utf-8')

    for i, text in enumerate(f.readlines()):
        text = text['text']
        tokenized = tokenizer(text, max_length=max_length, 
            pad_to_max_length=pad_to_max_length, return_tensors=return_tensors)

        if max_examples and i >= max_examples:
            break
        examples.append(tokenized)

    f.close()
    return examples

# TODO: parallelize
    # for i in range(len(text['text'])):
    #     sent = text['text'][i]
    #     doc = pos_tagger(sent)
    #     token_tags = {'tokens':[], 'tags':[]}
    #     letter_tags = []
    #     for token in doc:
    #         token_tags['tokens'].append(token.text)
    #         token_tags['tags'].append(token.pos_)
    #         for letter in token.text:
    #             #if letter not in ["�"]:
    #             letter_tags.append(token.pos_)

    #     hf_tokens_0 = tokenizer.tokenize(sent)
    #     hf_tokens = []
    #     for hf_token in hf_tokens_0:
    #         if hf_token[0] == "▁" and len(hf_token) > 1:
    #             hf_token = hf_token[1:]
    #             hf_tokens.append(hf_token)
    #         elif hf_token[0] == "▁" and len(hf_token) == 1:
    #             hf_tokens.append("")
    #         else:
    #             hf_tokens.append(hf_token)

    #     hf_letter_ids = []
    #     j = 0
    #     for hf_token in hf_tokens:
    #         hf_letter_id = []
    #         for hf_letter in hf_token:
    #             hf_letter_id.append(j)
    #             j+=1
    #         hf_letter_ids.append(hf_letter_id)

    #     # if len(letter_tags) - 1 != hf_letter_ids[-1][-1]:
    #     #     print(len(sent))
    #     #     print(sent)

    #     #     sent_2 = "".join(hf_tokens_0)
    #     #     print(len(sent_2))
    #     #     print(sent_2)

    #     #     sent_3 = "".join(token_tags['tokens'])
    #     #     print(len(sent_3))
    #     #     print(len(letter_tags))
    #     #     print(len(letter_tags) + sent_2.count("▁"))
    #     #     print(token_tags['tokens'])

    #     hf_tags = []
    #     for word in hf_letter_ids:
    #         tag_word = []
    #         for letter_idx in word:
    #             # print(letter_tags)
    #             # print(hf_letter_ids)
    #             # print(len(letter_tags))
    #             # print(word)
    #             # print(letter_idx)
    #             tag_word.append(letter_tags[letter_idx])
    #         if len(tag_word) > 0:
    #             hf_tags.append(tag_word[0])
    #             #assert (len(tag_word) == tag_word.count(tag_word[0]))
    #             # if len(tag_word) != tag_word.count(tag_word[0]):
    #             #     print(tag_word)
    #         else:
    #             hf_tags.append('NONE')

    #     noun_indicator = torch.tensor([pos_tag == 'NOUN' for pos_tag in hf_tags])
    #     # print(len(noun_indicator))
    #     # print(len(hf_tokens_0))

    # noun_indicator = torch.tensor([pos_tag == 'NOUN' for pos_tag in hf_tags])
    # tokenized['arg_ind'] = noun_indicator

"""
class LanguageModelingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./data/c4/en",
        split="train",
        max_length=1024,
        mlm_prob=0.15,
        tokenized=False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized = tokenized
        self.mlm_prob = mlm_prob
        self.split = split

        list_ids = {'train': [], 'val': []}
        train_files = []
        val_files = []
        for f in os.listdir(data_dir):
            file_path = os.path.join(data_dir, f)
            if os.path.isfile(file_path) and 'train' in str(file_path) and os.path.get_size(file_path) > 1000000:
                train_files.append(file_path)
            elif os.path.isfile(file_path) and 'validation' in str(file_path) and os.path.get_size(file_path) > 1000000:
                val_files.append(file_path)
        list_ids['train'] = files
        list_ids['val'] = files
        self.list_ids = list_ids

        c4_train = load_dataset('allenai/c4', data_files='en/c4-train.000**-of-01024.json.gz', streaming=True)
        tokenized_dataset = c4_train.map(encode_file)

        # if not tokenized:
        #     self.loaded = []
        #     for file_path in files:
        #         self.loaded += encode_and_mask_file(tokenizer, file_path, max_length)
        # else: #TODO   
        #     self.dataset = torch.load(os.path.join(data_dir, type_path))

    def __iter__(self):
        pass

    def __len__(self):
        # if self.tokenized:
        #     return len(self.dataset)
        # else:
        #     return len(self.loaded)
        return len(self.list_ids[self.split])

    def __getitem__(self, index):
        # if self.tokenized:
        #     dp = self.dataset[index]
        #     source_ids, src_mask, target_ids = dp[0], dp[1], dp[2]
        #     source_ids = source_ids[:self.max_source_length]
        #     src_mask = src_mask[:self.max_source_length]
        #     target_ids = target_ids[:self.max_target_length]
        # else:
        #     source_ids = self.source[index]["input_ids"].squeeze()
        #     src_mask = self.source[index]["attention_mask"].squeeze()

        file_path = self.list_ids[self.split][index]

        return {"input_ids": source_ids, "attention_mask": src_mask}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id, trim_y=True):
        if trim_y:
            y = trim_batch(batch["target_ids"], pad_token_id)
        else:
            y = batch["target_ids"]
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        # TODO: use .map to do this in parallel
        rand = torch.rand(tokenized.input_ids.shape)
        mask_arr = (rand < mlm_prob) * (inputs.input_ids != 1) * (inputs.input_ids != 0)
        selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
        tokenized.input_ids[0, selection] = 32099 #<extra_id_0>
        labels = tokenized.input_ids.detach().clone()
        tokenized['labels'] = labels

        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}


def worker_fn(dataset, index_queue, output_queue):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


class MultiProcessingDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=64,
        num_workers=1,
        prefetch_batches=2,
        collate_fn=default_collate,
    ):
        super().__init__(dataset, batch_size, collate_fn)

        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.dataset, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self):
        while (
            self.prefetch_index < len(self.dataset)
            and self.prefetch_index
            < self.index + 2 * self.num_workers * self.batch_size
        ):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def get(self):
        self.prefetch()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data

        self.index += 1
        return item

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
"""
