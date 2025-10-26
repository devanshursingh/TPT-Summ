from decomp import UDSCorpus
import json

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--output-json-1', help='path to write output json with list of json objects with sentence data to', default='uds_sentences.json')
parser.add_argument('--output-json-2', help='path to write output json with list of protorole vectors to', default='uds_protoroles.json')

args = parser.parse_args()

# read the specified split of the UDS corpus
uds_train = UDSCorpus(split='train')
uds_dev = UDSCorpus(split='dev')
uds_test = UDSCorpus(split='test')

# query the dataset for all semantic dependency edges
edge_results_train = {}
for gid, graph in uds_train.items():
    edge_results_train.update({gid: graph.semantics_edges()})
edge_results_dev = {}
for gid, graph in uds_dev.items():
    edge_results_dev.update({gid: graph.semantics_edges()})
edge_results_test = {}
for gid, graph in uds_test.items():
    edge_results_test.update({gid: graph.semantics_edges()})

training_examples = []
protorole_dataset = {'protoroles': []}

empty_protorole_vector = [0] * 14 #14 is number of protorole attributes

for gid, semantics in edge_results_train.items():
    if bool(semantics):
        sent_id = gid
        sent = uds_train[gid].sentence
        sent_tokenized = sent.split()

        sent_token_slices = []
        i = 0
        for token in sent_tokenized:
            sent_token_slices.append([i, i+len(token)])
            i = i + len(token) + 1
        # [seq_size, 14]
        protoroles_token = [empty_protorole_vector] * len(sent_tokenized)
        protoroles_char = [empty_protorole_vector] * len(sent)
        #arg_ind = [0] * len(sent_tokenized)
        arg_ind = [0] * len(sent)
        for nodes, annotations in semantics.items():
            if 'protoroles' in annotations.keys():
                pred_id, arg_id = nodes
                word_idx = int(arg_id.split("-")[-1]) - 1
                start_idx = sent_token_slices[word_idx][0]
                stop_idx = sent_token_slices[word_idx][1]
                #char_slice = slice(start_idx, stop_idx, 1)

                protorole_annotations = annotations.get('protoroles')
                protorole_vector = []
                for attr, val in protorole_annotations.items():
                    value = val.get('value')
                    protorole_vector.append(value)

                protoroles_token[word_idx] = protorole_vector
                #arg_ind[word_idx] = 1

                for i in range(start_idx, stop_idx):
                    arg_ind[i] = 1
                    protoroles_char[i] = protorole_vector

        if sum(arg_ind) > 0:
            x = 0
            for vector in protoroles_token:
                if len(vector) == 14:
                    x+=1
            if x == len(protoroles_token):
                # create training example
                training_examples.append({
                    'sent_id': sent_id,
                    'sentence': sent,
                    'protoroles_token': protoroles_token,
                    'protoroles_char': protoroles_char,
                    'arg_ind': arg_ind,
                })

                for pr_vec in protoroles_token:
                    if pr_vec != empty_protorole_vector:
                        protorole_dataset['protoroles'].append(protorole_vector)

for gid, semantics in edge_results_dev.items():
    if bool(semantics):
        sent_id = gid
        sent = uds_dev[gid].sentence
        sent_tokenized = sent.split()

        sent_token_slices = []
        i = 0
        for token in sent_tokenized:
            sent_token_slices.append([i, i+len(token)])
            i = i + len(token) + 1

        empty_protorole_vector = [0] * 14 #14 is number of protorole attributes
        # [seq_size, 14]
        protoroles_token = [empty_protorole_vector] * len(sent_tokenized)
        protoroles_char = [empty_protorole_vector] * len(sent)
        #arg_ind = [0] * len(sent_tokenized)
        arg_ind = [0] * len(sent)
        for nodes, annotations in semantics.items():
            if 'protoroles' in annotations.keys():
                pred_id, arg_id = nodes
                word_idx = int(arg_id.split("-")[-1]) - 1
                start_idx = sent_token_slices[word_idx][0]
                stop_idx = sent_token_slices[word_idx][1]
                #char_slice = slice(start_idx, stop_idx, 1)

                protorole_annotations = annotations.get('protoroles')
                protorole_vector = []
                for attr, val in protorole_annotations.items():
                    value = val.get('value')
                    protorole_vector.append(value)
                
                protoroles_token[word_idx] = protorole_vector
                #arg_ind[word_idx] = 1

                for i in range(start_idx, stop_idx):
                    arg_ind[i] = 1
                    protoroles_char[i] = protorole_vector

        if sum(arg_ind) > 0:
            x = 0
            for vector in protoroles_token:
                if len(vector) == 14:
                    x+=1
            if x == len(protoroles_token):
                # create training example
                training_examples.append({
                    'sent_id': sent_id,
                    'sentence': sent,
                    'protoroles_token': protoroles_token,
                    'protoroles_char': protoroles_char,
                    'arg_ind': arg_ind,
                })

                for pr_vec in protoroles_token:
                    if pr_vec != empty_protorole_vector:
                        protorole_dataset['protoroles'].append(protorole_vector)

for gid, semantics in edge_results_test.items():
    if bool(semantics):
        sent_id = gid
        sent = uds_test[gid].sentence
        sent_tokenized = sent.split()

        sent_token_slices = []
        i = 0
        for token in sent_tokenized:
            sent_token_slices.append([i, i+len(token)])
            i = i + len(token) + 1

        empty_protorole_vector = [0] * 14 #14 is number of protorole attributes
        # [seq_size, 14]
        protoroles_token = [empty_protorole_vector] * len(sent_tokenized)
        protoroles_char = [empty_protorole_vector] * len(sent)
        #arg_ind = [0] * len(sent_tokenized)
        arg_ind = [0] * len(sent)
        for nodes, annotations in semantics.items():
            if 'protoroles' in annotations.keys():
                pred_id, arg_id = nodes
                word_idx = int(arg_id.split("-")[-1]) - 1
                start_idx = sent_token_slices[word_idx][0]
                stop_idx = sent_token_slices[word_idx][1]
                #char_slice = slice(start_idx, stop_idx, 1)

                protorole_annotations = annotations.get('protoroles')
                protorole_vector = []
                for attr, val in protorole_annotations.items():
                    value = val.get('value')
                    protorole_vector.append(value)
                
                protoroles_token[word_idx] = protorole_vector
                #arg_ind[word_idx] = 1

                for i in range(start_idx, stop_idx):
                    arg_ind[i] = 1
                    protoroles_char[i] = protorole_vector

        if sum(arg_ind) > 0:
            x = 0
            for vector in protoroles_token:
                if len(vector) == 14:
                    x+=1
            if x == len(protoroles_token):
                # create training example
                training_examples.append({
                    'sent_id': sent_id,
                    'sentence': sent,
                    'protoroles_token': protoroles_token,
                    'protoroles_char': protoroles_char,
                    'arg_ind': arg_ind,
                })

                for pr_vec in protoroles_token:
                    if pr_vec != empty_protorole_vector:
                        protorole_dataset['protoroles'].append(protorole_vector)

print(len(training_examples))
print(len(protorole_dataset['protoroles']))

with open(args.output_json_1, "w") as f:
    json.dump({"data": training_examples}, f, indent=2)

with open(args.output_json_2, "w") as f:
    json.dump(protorole_dataset, f, indent=2)