import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def find_entity(source: List[int], target: List[int]) -> int:
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def to_tuple(sent: dict):
    """ list to tuple (inplace operation)
    """
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list

def filter_data(fpath: str, rel2id: Dict):
    filtered_data = []
    for obj in json.load(open(fpath)):
        filtered_triples = []
        if 'NYT11-HRL' in fpath and len(obj['triple_list']) != 1:
            continue
        for triple in obj['triple_list']:
            if triple[1] not in rel2id:
                continue
            filtered_triples.append(triple)
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
        print("train text insight")
        print(f" max len: {max(text_lens)}")
        print(f" min len: {min(text_lens)}")
        print(f" avg len: {sum(text_lens) / len(text_lens)}")
    for sent in data:
        to_tuple(sent)
    print(f"data len: {len(data)}")
    return data

class TDEERDataset(Dataset):
    def __init__(self, datas: List, tokenizer: BertTokenizer, rel2id: Dict, all_rels: List, max_len: int,
                 max_sample_triples: Optional[int] = None, neg_samples: Optional[int] = None):
        self.max_sample_triples = max_sample_triples
        self.neg_samples = neg_samples
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

            text_tokened = tokenizer(data['text'], max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
            entity_set = set()  # (head idx, tail idx)
            triples_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            subj_rel_set = set()   # (sub head, sub tail, rel)
            subj_set = set()   # (sub head, sub tail)
            rel_set = set()
            trans_map = defaultdict(list)   # {(sub_head, rel): [tail_heads]}
            for triple in data['triple_list']:
                subj, rel, obj = triple
                rel_idx = self.rel2id[rel]
                subj_tokened = tokenizer(subj, add_special_tokens=False)
                obj_tokened = tokenizer(obj, add_special_tokens=False)
                subj_head_idx = find_entity(text_tokened.input_ids[0].tolist(), subj_tokened.input_ids)
                subj_tail_idx = subj_head_idx + len(subj_tokened.input_ids) - 1
                obj_head_idx = find_entity(text_tokened.input_ids[0].tolist(), obj_tokened.input_ids)
                obj_tail_idx = obj_head_idx + len(obj_tokened.input_ids) - 1
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

            entity_heads = torch.zeros((self.max_len, 2))
            entity_tails = torch.zeros((self.max_len, 2))
            for (head, tail, _type) in entity_set:
                entity_heads[head][_type] = 1
                entity_tails[tail][_type] = 1

            rels = torch.zeros(self.relation_size)
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
                sample_obj_heads = torch.zeros(self.max_len)
                for idx in trans_map[(subj_head_idx, subj_tail_idx, rel_idx)]:
                    sample_obj_heads[idx] = 1.0
                # positive samples
                pos_datas.append({
                    'token_ids': text_tokened.input_ids[0],
                    'attention_mask': text_tokened.attention_mask[0],
                    'entity_heads': entity_heads,
                    'entity_tails': entity_tails,
                    'rels': rels,
                    'sample_subj_head': torch.tensor(subj_head_idx, dtype=torch.long),
                    'sample_subj_tail': torch.tensor(subj_tail_idx, dtype=torch.long),
                    'sample_rel': torch.tensor(rel_idx, dtype=torch.long),
                    'sample_obj_heads': sample_obj_heads,
                })

                # Generate negative samples
                if self.neg_samples:
                    for _ in range(self.neg_samples):
                        neg_rel = np.random.choice(self.rels_set)
                        if neg_rel == rel_idx:
                            continue
                        if (subj_head_idx, subj_tail_idx, neg_rel) in neg_history:
                            continue
                        neg_history.add((subj_head_idx, subj_tail_idx, neg_rel))
                        
                        neg_sample_obj_heads = torch.zeros(self.max_len)
                        for idx in trans_map.get((subj_head_idx, subj_tail_idx, neg_rel), []):
                            neg_sample_obj_heads[idx] = 1.0
                        
                        current_neg_datas.append({
                            'token_ids': text_tokened.input_ids[0],
                            'attention_mask': text_tokened.attention_mask[0],
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': torch.tensor(subj_head_idx, dtype=torch.long),
                            'sample_subj_tail': torch.tensor(subj_tail_idx, dtype=torch.long),
                            'sample_rel': torch.tensor(neg_rel, dtype=torch.long),
                            'sample_obj_heads': neg_sample_obj_heads,
                        })

                neg_datas.extend(current_neg_datas)

            current_datas = pos_datas + neg_datas
            self.datas.extend(current_datas)

        print(f"Total number of samples: {len(self.datas)}")
        print(f"Number of positive samples: {len(pos_datas)}")
        print(f"Number of negative samples: {len(neg_datas)}")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        item = self.datas[idx]
        required_keys = ['token_ids', 'attention_mask', 'entity_heads', 'entity_tails', 'rels', 'sample_subj_head', 'sample_subj_tail', 'sample_rel', 'sample_obj_heads']
        for key in required_keys:
            assert key in item, f"Required key '{key}' not found in item at index {idx}"
        return item

def collate_fn(batch):
    def stack_or_pad(key):
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        elif isinstance(values[0], (int, np.int64)):
            return torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], list):
            max_len = max(len(v) for v in values)
            return torch.tensor([v + [0] * (max_len - len(v)) for v in values])
        else:
            print(f"Warning: Unexpected type in collate_fn for {key}: {type(values[0])}")
            return values

    result = {}
    for key in batch[0].keys():
        try:
            result[key] = stack_or_pad(key)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            result[key] = [d[key] for d in batch]  # 退回到简单的列表
    
    # 确保 'token_ids' 存在
    if 'token_ids' not in result:
        print("Warning: 'token_ids' not found in batch. Keys present:", result.keys())
        if 'input_ids' in result:
            result['token_ids'] = result['input_ids']
            print("Using 'input_ids' as 'token_ids'")
    
    return result

def get_dataloader(dataset: TDEERDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)