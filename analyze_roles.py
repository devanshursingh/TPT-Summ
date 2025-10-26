"""
Decode summarization models (generate summaries) trained with finetune.py
Multi-GPU decoding not working yet.
"""

import argparse
import os
import logging
import glob
from pathlib import Path
from unicodedata import name
import numpy as np
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from rouge import Rouge
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer, T5TokenizerFast
from models.modeling_tpt import TPTForConditionalGeneration, TPTWithLMHeadModel
from models.configuration_tpt import TPTConfig
from uds_dataset import UDSDataset

logger = logging.getLogger(__name__)

MODELS = {
    "t5": T5ForConditionalGeneration,
    "tp": TPTForConditionalGeneration,
    "tp-mlm": TPTWithLMHeadModel,
}

CONFIGS = {
    "t5": T5Config,
    "tp-mlm": TPTConfig,
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(args):
    args.batch_size = args.batch_size * max(1, args.n_gpu)

    logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
    checkpoints = list(sorted(glob.glob(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"), recursive=True)))
    print(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"))
    checkpoint = checkpoints[0]

    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    num_epoch = checkpoint.split("epoch=")[1].split(".ckpt")[0]

    # Reload the model
    config = CONFIGS["tp-mlm"].from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    config.num_roles = 50
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     cache_dir=args.cache_dir,
    # )
    tokenizer = T5TokenizerFast.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    # model = MODELS[args.model_name_or_path[:2]].from_pretrained(
    #     args.model_name_or_path,
    #     return_unpretrained=True,
    #     config=config,
    #     cache_dir=args.cache_dir,
    # )
    model = MODELS["tp-mlm"](config=config)

    # Restore the model parameters, TODO: check that this works
    state_dict_pl = torch.load(checkpoint, map_location=torch.device('cpu'))["state_dict"]
    state_dict = {}
    for weight in state_dict_pl:
        if 'label_smoothing' in weight or 'rq' in weight or 'rk' in weight:
            continue
        if weight.startswith("model."):
            state_dict[weight[6:]] = state_dict_pl[weight]
        else:
            state_dict[weight] = state_dict_pl[weight]
    model.load_state_dict(state_dict)
    model.to(args.device)

    dataset = UDSDataset(tokenizer=tokenizer, data_file=args.data_file)
    print("Number of sentences with protoroles:")
    print(len(dataset))
    eval_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("masked_language_modeling", {}))

    role_dataset_concat = {'protoroles': [], 'roles': [], 'role_matrix': []}
    role_dataset_head = {'protoroles': [], 'roles': [], 'role_matrix': []}
    role_dataset_head1 = {'protoroles': [], 'roles': [], 'role_matrix': []}
    role_dataset_head2 = {'protoroles': [], 'roles': [], 'role_matrix': []}
    role_dataset_head3 = {'protoroles': [], 'roles': [], 'role_matrix': []}
    analysis_dataset = {'sentences': [], 'tokens': [], 'role_id': [], 'role_attention': [], 'arg_ind': [], 'protoroles': []}
    
    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        # [batch_size, seq_size]
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        # [batch_size, seq_size]
        arg_ind = batch["arg_ind"]
        # [batch_size, seq_size, 14]
        protoroles = batch["protoroles"]

        #batch = tuple(batch[t].to(args.device) for t in batch)
        #input_ids, attention_mask = batch[0], batch[1]
        enc_self_attn = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[-1]
        input_ids = input_ids.data.cpu()

        # is this [batch_size, n_heads, seq_size, num_roles]
        enc_self_role_attention = model.encoder.block[-1].layer[0].SelfAttention.role_weights.data.cpu()
        # [num_roles, role_dim]
        Roles = model.encoder.block[-1].layer[0].SelfAttention.R
        # TODO: normalize role matrix
        Roles = Roles / torch.norm(Roles, dim=1, keepdim=True)
        Roles = Roles.data.cpu()
        
        # [batch_size, n_heads, seq_size]
        encoder_max_role_attention, encoder_role_predictions = torch.max(enc_self_attn[:,-1,:,:,:], dim=3)
        # [batch_size, seq_size, n_heads]
        encoder_max_role_attention = encoder_max_role_attention.permute(0,2,1)
        encoder_role_predictions = encoder_role_predictions.permute(0,2,1)

        arg_mask = arg_ind[:, :, None, None].permute(0, 2, 1, 3)
        # [batch_size, n_heads, seq_size, num_roles]? Some seq_size entries will be 0 all the way down
        role_weights = enc_self_role_attention * arg_mask
        role_weights = role_weights[:, :, :, :, None]
        Roles = Roles[None, None, None, :, :]
        # [batch_size, n_heads, seq_size, num_roles, 1] dot [1, 1, 1, num_roles, role_dim]
        # [batch_size, n_heads, seq_size, role_dim]
        arg_roles = torch.sum(role_weights * Roles, dim=3)

        # protoroles: [batch_size, seq_size, 14]
        # roles: [batch_size, n_heads, seq_size, role_dim]
        for k in range(arg_roles.shape[0]):
            indices = [i for i, e in enumerate(arg_ind[k].tolist()) if e == 1]
            for r in indices:
                # [n_heads, role_dim]
                # arg_roles[k, :, r, :]
                # [14]
                # protoroles[k, r, :]
                # [n_heads * role_dim] concatenate all heads
                t = torch.reshape(arg_roles[k, :, r, :], (-1,))
                role_dataset_concat['protoroles'].append(protoroles[k, r, :].tolist())
                role_dataset_concat['roles'].append(t.tolist())
                # separate entries for each head or pick one head
                for h in range(arg_roles.shape[1]):
                    # [role_dim]
                    role_dataset_head['roles'].append(arg_roles[k, h, r, :].tolist())
                    # [14]
                    role_dataset_head['protoroles'].append(protoroles[k, r, :].tolist())

                # [role_dim]
                role_dataset_head1['roles'].append(arg_roles[k, 0, r, :].tolist())
                # [14]
                role_dataset_head1['protoroles'].append(protoroles[k, r, :].tolist())

                # [role_dim]
                role_dataset_head2['roles'].append(arg_roles[k, 1, r, :].tolist())
                # [14]
                role_dataset_head2['protoroles'].append(protoroles[k, r, :].tolist())

                # [role_dim]
                role_dataset_head3['roles'].append(arg_roles[k, 2, r, :].tolist())
                # [14]
                role_dataset_head3['protoroles'].append(protoroles[k, r, :].tolist())

        for b in range(arg_ind.shape[0]):
            sent_input_ids = input_ids[b, :]
            sent_arg_ind = arg_ind[b, :].tolist()
            sent_protoroles = protoroles[b, :, :].tolist()
            sent_max_role_ind = encoder_role_predictions[b, :, :].tolist()
            sent_max_role_attention = encoder_max_role_attention[b, :, :].tolist()
            sent_text = tokenizer.decode(sent_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            sent_tokens = tokenizer.convert_ids_to_tokens(sent_input_ids, skip_special_tokens=True)

            analysis_dataset['sentences'].append(sent_text)
            analysis_dataset['tokens'].append(sent_tokens)
            analysis_dataset['role_id'].append(sent_max_role_ind)
            analysis_dataset['role_attention'].append(sent_max_role_attention)
            analysis_dataset['arg_ind'].append(sent_arg_ind)
            analysis_dataset['protoroles'].append(sent_protoroles)


    role_dataset_concat['role_matrix'] = Roles.tolist()
    role_dataset_head['role_matrix'] = Roles.tolist()
    role_dataset_head3['role_matrix'] = Roles.tolist()

    print("Concatenated head roles dataset length:")
    print(len(role_dataset_concat['roles']))

    print("Separate entry head roles dataset length:")
    print(len(role_dataset_head['roles']))
    
    print("Analysis dataset length:")
    print(len(analysis_dataset['arg_ind']))

    # name1 = 'role_dataset_concat_' + str(args.run_id) + '.json'
    # with open(name1, "w") as f:
    #     json.dump(role_dataset_concat, f, indent=2)

    # name2 = 'role_dataset_head_' + str(args.run_id) + '.json'
    # with open(name2, "w") as f:
    #     json.dump(role_dataset_head, f, indent=2)

    name3 = 'analysis_dataset_' + str(args.run_id) + '.json'
    with open(name3, "w") as f:
        json.dump(analysis_dataset, f, indent=2)

    # name4 = 'role_dataset_head_3_' + str(args.run_id) + '.json'
    # with open(name4, "w") as f:
    #     json.dump(role_dataset_head3, f, indent=2)
    
    # name5 = 'role_dataset_head_1_' + str(args.run_id) + '.json'
    # with open(name5, "w") as f:
    #     json.dump(role_dataset_head1, f, indent=2)

    # name6 = 'role_dataset_head_2_' + str(args.run_id) + '.json'
    # with open(name6, "w") as f:
    #     json.dump(role_dataset_head2, f, indent=2)


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="",
    )
    parser.add_argument(
        "--data_file", type=str, default="uds_train.json", help="",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="c4", help="dataset that model was trained on",
    )
    parser.add_argument(
        "--model_path", type=str, default="out", help="the location of the model to be eval.",
    )
    parser.add_argument(
        "--run_id", type=str, default='00', help="",
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=-1, help="",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.model_path = os.path.join(args.model_path, args.dataset_name, args.run_id)
    generate_summaries(args)


if __name__ == "__main__":
    run_generate()