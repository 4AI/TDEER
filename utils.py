import json
import time
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

def rematch(offsets: List) -> List:
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping

class Infer:
    def __init__(self, model: torch.nn.Module, tokenizer, id2rel: Dict, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.device = device

    def decode_entity(self, text: str, mapping: List, start: int, end: int):
        s = mapping[start]
        e = mapping[end]
        s = 0 if not s else s[0]
        e = len(text) - 1 if not e else e[-1]
        entity = text[s: e + 1]
        return entity

    def __call__(self, text: str, threshold: float = 0.01) -> Set:  # MODIFIED: changed threshold from 0.1 to 0.01
        self.model.eval()
        with torch.no_grad():
            tokened = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512, return_offsets_mapping=True)
            tokened = {k: v.to(self.device) for k, v in tokened.items()}
            
            entity_heads_logits, entity_tails_logits, relations_logits = self.model(tokened['input_ids'], tokened['attention_mask'])

            # print(f"Max entity_heads_logits: {entity_heads_logits.max().item()}")
            # print(f"Max entity_tails_logits: {entity_tails_logits.max().item()}")
            # print(f"Max relations_logits: {relations_logits.max().item()}")
            entity_heads_probs = torch.sigmoid(entity_heads_logits)
            entity_tails_probs = torch.sigmoid(entity_tails_logits)
            relations_probs = torch.sigmoid(relations_logits)

            entity_heads = torch.where(entity_heads_logits[0] > threshold)
            entity_tails = torch.where(entity_tails_logits[0] > threshold)
            relations = torch.where(relations_logits[0] > threshold)[0].tolist()

            # print(f"Number of entity heads: {len(entity_heads[0])}")
            # print(f"Number of entity tails: {len(entity_tails[0])}")
            # print(f"Number of relations: {len(relations)}")

            subjects = []
            entity_map = {}
            for head, head_type in zip(*entity_heads):
                for tail, tail_type in zip(*entity_tails):
                    if head <= tail and head_type == tail_type:
                        entity = self.decode_entity(text, tokened['offset_mapping'][0].tolist(), head, tail)
                        if head_type == 0:
                            subjects.append((entity, head.item(), tail.item()))
                        else:
                            entity_map[head.item()] = entity
                        break

            # print(f"Number of subjects: {len(subjects)}")
            # print(f"Number of entities in entity_map: {len(entity_map)}")

            triple_set = set()
            if subjects and relations:
                for (sub, sub_head, sub_tail) in subjects:
                    for rel in relations:
                        sub_head_tensor = torch.tensor([sub_head], device=self.device)
                        sub_tail_tensor = torch.tensor([sub_tail], device=self.device)
                        rel_tensor = torch.tensor([rel], device=self.device)
                        
                        _, _, _, obj_head_logits = self.model(
                            tokened['input_ids'], 
                            tokened['attention_mask'],
                            sample_subj_head=sub_head_tensor,
                            sample_subj_tail=sub_tail_tensor,
                            sample_rel=rel_tensor
                        )
                        
                        # print(f"Max obj_head_logits: {obj_head_logits.max().item()}")
                        # print(f"Min obj_head_logits: {obj_head_logits.min().item()}")
                        # print(f"Mean obj_head_logits: {obj_head_logits.mean().item()}")
                        
                        for h in torch.where(obj_head_logits[0] > threshold)[0].tolist():  # MODIFIED: changed threshold from 0.1 to 0.01
                            if h in entity_map:
                                obj = entity_map[h]
                                triple_set.add((sub, self.id2rel[rel], obj))
                            # print(f"Predicted object head: {h}")
                            # print(f"Object in entity_map: {h in entity_map}")

            print(f"Number of triples: {len(triple_set)}")
            print(f"Triples: {triple_set}")
        
        return triple_set

def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold

def remove_space(data_set):
    data_set = {(i[0].replace(' ', ''), i[1], i[2].replace(' ', '')) for i in data_set}
    return data_set

def compute_metrics(infer, dev_data, exact_match=False, model_name='tmp'):
    output_path = f'{model_name}.output'
    if output_path:
        writer = open(output_path, 'w')
    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    infer_times = []
    for line in tqdm(iter(dev_data)):
        start_time = time.time()
        pred_triples = infer(line['text'])
        infer_times.append(time.time() - start_time)
        gold_triples = set(line['triple_list'])

        if exact_match:
            gold_triples = remove_space(gold_triples)
            pred_triples = remove_space(pred_triples)

        pred_triples_eval, gold_triples_eval = partial_match(pred_triples, gold_triples) if not exact_match else (pred_triples, gold_triples)

        correct_num += len(pred_triples_eval & gold_triples_eval)
        predict_num += len(pred_triples_eval)
        gold_num += len(gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'golds': [
                    dict(zip(orders, triple)) for triple in gold_triples
                ],
                'preds': [
                    dict(zip(orders, triple)) for triple in pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                ]
            }, ensure_ascii=False)
            writer.write(result + '\n')
    if output_path:
        writer.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
    print("avg infer time:", sum(infer_times) / len(infer_times))
    return precision, recall, f1_score