# -*- coding:utf-8 -*-

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import log

def find_entity(source, target):
    """ Find the start index of the target sequence in the source sequence """
    target_len = len(target)
    for i in range(len(source) - target_len + 1):
        if source[i:i+target_len] == target:
            return i
    return -1

def to_tuple(sent):
    """ Convert lists to tuples in place """
    sent['triple_list'] = [tuple(triple) for triple in sent['triple_list']]

def filter_data(fpath: str, rel2id: dict):
    filtered_data = []
    try:
        with open(fpath, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("File not found:", fpath)
        return filtered_data
    except json.JSONDecodeError:
        print("Error decoding JSON from:", fpath)
        return filtered_data

    for obj in data:
        if 'NYT11-HRL' in fpath and len(obj.get('triple_list', [])) != 1:
            continue
        filtered_triples = [triple for triple in obj.get('triple_list', []) if triple[1] in rel2id]
        if not filtered_triples:
            continue
        obj['triple_list'] = filtered_triples
        filtered_data.append(obj)
    
    return filtered_data

def load_rel(rel_path: str) -> Tuple[Dict, Dict, List]:
    id2rel, rel2id = json.load(open(rel_path))
    all_rels = list(id2rel.keys())
    id2rel = {int(i): j for i, j in id2rel.items()}
    return id2rel, rel2id, all_rels


def load_data(fpath: str, rel2id: Dict, is_train: bool = False) -> List:
    data = filter_data(fpath, rel2id)
    if is_train:
        text_lens = [len(obj['text'].split()) for obj in data]
        log.info("train text insight")
        log.info(f" max len: {max(text_lens)}")
        log.info(f" min len: {min(text_lens)}")
        log.info(f" avg len: {sum(text_lens) / len(text_lens)}")
    for sent in data:
        to_tuple(sent)
    log.info(f"data len: {len(data)}")
    return data

class DataGenerator(Dataset):
    def __init__(self, datas: List, tokenizer: BertTokenizer, rel2id: Dict, all_rels: List, max_len: int,
                 batch_size: int = 32, max_sample_triples: Optional[int] = None, neg_samples: Optional[int] = None):
        self.max_sample_triples = max_sample_triples
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rel2id = rel2id
        self.rels_set = list(rel2id.values())
        self.relation_size = len(rel2id)
        self.num_rels = len(all_rels)
        self.all_rels = all_rels

        self.datas = []

        for data in datas:
            pos_datas = []
            neg_datas = []

            text_tokened = tokenizer.encode_plus(data['text'], truncation=True, padding='max_length', max_length=max_len)
            entity_set = set()  # (head idx, tail idx)
            triples_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            subj_rel_set = set()   # (sub head, sub tail, rel)
            subj_set = set()   # (sub head, sub tail)
            rel_set = set()
            trans_map = defaultdict(list)   # {(sub_head, rel): [tail_heads]}
            for triple in data['triple_list']:
                subj, rel, obj = triple
                rel_idx = self.rel2id[rel]
                subj_tokened = tokenizer.encode_plus(subj, add_special_tokens=False)
                obj_tokened = tokenizer.encode_plus(obj, add_special_tokens=False)
                subj_head_idx = find_entity(text_tokened['input_ids'], subj_tokened['input_ids'])
                subj_tail_idx = subj_head_idx + len(subj_tokened['input_ids']) - 1
                obj_head_idx = find_entity(text_tokened['input_ids'], obj_tokened['input_ids'])
                obj_tail_idx = obj_head_idx + len(obj_tokened['input_ids']) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                entity_set.add((subj_head_idx, subj_tail_idx, 0))
                entity_set.add((obj_head_idx, obj_tail_idx, 1))
                subj_rel_set.add((subj_head_idx, subj_tail_idx, rel_idx))
                subj_set.add((subj_head_idx, subj_tail_idx))
                triples_set.add(
                    (subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx)
                )
                rel_set.add(rel_idx)
                trans_map[(subj_head_idx, subj_tail_idx, rel_idx)].append(obj_head_idx)

            if not rel_set:
                continue

            entity_heads = np.zeros((self.max_len, 2))
            entity_tails = np.zeros((self.max_len, 2))
            for (head, tail, _type) in entity_set:
                entity_heads[head][_type] = 1
                entity_tails[tail][_type] = 1

            rels = np.zeros(self.relation_size)
            for idx in rel_set:
                rels[idx] = 1

            if self.max_sample_triples is not None:
                triples_list = list(triples_set)
                np.random.shuffle(triples_list)
                triples_list = triples_list[:self.max_sample_triples]
            else:
                triples_list = list(triples_set)

            neg_history = set()
            for subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx in triples_list:
                current_neg_datas = []
                sample_obj_heads = np.zeros(self.max_len)
                for idx in trans_map[(subj_head_idx, subj_tail_idx, rel_idx)]:
                    sample_obj_heads[idx] = 1.0
                # postive samples
                pos_datas.append({
                    'token_ids': text_tokened['input_ids'],
                    'segment_ids': text_tokened['token_type_ids'],
                    'entity_heads': entity_heads,
                    'entity_tails': entity_tails,
                    'rels': rels,
                    'sample_subj_head': subj_head_idx,
                    'sample_subj_tail': subj_tail_idx,
                    'sample_rel': rel_idx,
                    'sample_obj_heads': sample_obj_heads,
                })

                # 1. inverse (tail as subj)
                neg_subj_head_idx = obj_head_idx
                neg_sub_tail_idx = obj_tail_idx
                neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                    current_neg_datas.append({
                        'token_ids': text_tokened['input_ids'],
                        'segment_ids': text_tokened['token_type_ids'],
                        'entity_heads': entity_heads,
                        'entity_tails': entity_tails,
                        'rels': rels,
                        'sample_subj_head': neg_subj_head_idx,
                        'sample_subj_tail': neg_sub_tail_idx,
                        'sample_rel': rel_idx,
                        'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                    })
                    neg_history.add(neg_pair)

                # 2. (pos sub, neg_rel)
                for neg_rel_idx in rel_set - {rel_idx}:
                    neg_pair = (subj_head_idx, subj_tail_idx, neg_rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            'token_ids': text_tokened['input_ids'],
                            'segment_ids': text_tokened['token_type_ids'],
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': subj_head_idx,
                            'sample_subj_tail': subj_tail_idx,
                            'sample_rel': neg_rel_idx,
                            'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                        })
                        neg_history.add(neg_pair)

                # 3. (neg sub, pos rel)
                for (neg_subj_head_idx, neg_sub_tail_idx) in subj_set - {(subj_head_idx, subj_tail_idx)}:
                    neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            'token_ids': text_tokened['input_ids'],
                            'segment_ids': text_tokened['token_type_ids'],
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': neg_subj_head_idx,
                            'sample_subj_tail': neg_sub_tail_idx,
                            'sample_rel': rel_idx,
                            'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                        })
                        neg_history.add(neg_pair)

                np.random.shuffle(current_neg_datas)
                if self.neg_samples is not None:
                    current_neg_datas = current_neg_datas[:self.neg_samples]
                neg_datas += current_neg_datas
            current_datas = pos_datas + neg_datas
            self.datas.extend(current_datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        sample = self.datas[idx]
    
        return {
            'token_ids': sample['token_ids'],
            'segment_ids': sample['segment_ids'],
            'entity_heads': sample['entity_heads'],
            'entity_tails': sample['entity_tails'],
            'rels': sample['rels'],
            'sample_subj_head': sample['sample_subj_head'],
            'sample_subj_tail': sample['sample_subj_tail'],
            'sample_rel': sample['sample_rel'],
            'sample_obj_heads': sample['sample_obj_heads']
        }
        

def collate_fn(batch):
    print("Batch input:", batch)
    batch_tokens = [item['token_ids'] for item in batch]
    batch_attention_masks = [[1] * len(tokens) for tokens in batch_tokens]
    batch_segments = [item['segment_ids'] for item in batch]
    batch_entity_heads = [item['entity_heads'] for item in batch]
    batch_entity_tails = [item['entity_tails'] for item in batch]
    batch_rels = [item['rels'] for item in batch]
    batch_sample_subj_head = [item['sample_subj_head'] for item in batch]
    batch_sample_subj_tail = [item['sample_subj_tail'] for item in batch]
    batch_sample_rel = [item['sample_rel'] for item in batch]
    batch_sample_obj_heads = [item['sample_obj_heads'] for item in batch]

    # Convert lists to NumPy arrays
    batch_tokens_np = np.array(batch_tokens)
    batch_attention_masks_np = np.array(batch_attention_masks)
    batch_segments_np = np.array(batch_segments)
    batch_entity_heads_np = np.array(batch_entity_heads)
    batch_entity_tails_np = np.array(batch_entity_tails)
    batch_rels_np = np.array(batch_rels)
    batch_sample_subj_head_np = np.array(batch_sample_subj_head)
    batch_sample_subj_tail_np = np.array(batch_sample_subj_tail)
    batch_sample_rel_np = np.array(batch_sample_rel)
    batch_sample_obj_heads_np = np.array(batch_sample_obj_heads)

    # Convert NumPy arrays to PyTorch tensors
    batch_tokens = torch.tensor(batch_tokens_np, dtype=torch.long)
    batch_attention_masks = torch.tensor(batch_attention_masks_np, dtype=torch.long)
    batch_segments = torch.tensor(batch_segments_np, dtype=torch.long)
    batch_entity_heads = torch.tensor(batch_entity_heads_np, dtype=torch.float)
    batch_entity_tails = torch.tensor(batch_entity_tails_np, dtype=torch.float)
    batch_rels = torch.tensor(batch_rels_np, dtype=torch.float)
    batch_sample_subj_head = torch.tensor(batch_sample_subj_head_np, dtype=torch.long)
    batch_sample_subj_tail = torch.tensor(batch_sample_subj_tail_np, dtype=torch.long)
    batch_sample_rel = torch.tensor(batch_sample_rel_np, dtype=torch.long)
    batch_sample_obj_heads = torch.tensor(batch_sample_obj_heads_np, dtype=torch.float)

    return {
        'token_ids': batch_tokens,
        'attention_mask': batch_attention_masks,
        'segment_ids': batch_segments,
        'entity_heads': batch_entity_heads,
        'entity_tails': batch_entity_tails,
        'rels': batch_rels,
        'sample_subj_head': batch_sample_subj_head,
        'sample_subj_tail': batch_sample_subj_tail,
        'sample_rel': batch_sample_rel,
        'sample_obj_heads': batch_sample_obj_heads
    }