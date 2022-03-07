#! -*- coding:utf-8 -*-

import json
import time
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm
from langml.tensor_typing import Models


def rematch(offsets: List) -> List:
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append([i for i in range(offset[0], offset[1])])
    return mapping


class Infer:
    def __init__(self, entity_model: Models, rel_model: Models, translate_mdoel: Models,
                 tokenizer: object, id2rel: Dict):
        self.entity_model = entity_model
        self.rel_model = rel_model
        self.translate_model = translate_mdoel
        self.tokenizer = tokenizer
        self.id2rel = id2rel

    def decode_entity(self, text: str, mapping: List, start: int, end: int):
        s = mapping[start]
        e = mapping[end]
        s = 0 if not s else s[0]
        e = len(text) - 1 if not e else e[-1]
        entity = text[s: e + 1]
        return entity

    def __call__(self, text: str, threshold: float = 0.5) -> Set:
        tokened = self.tokenizer.encode(text)
        token_ids, segment_ids = np.array([tokened.ids]), np.array([tokened.type_ids])
        mapping = rematch(tokened.offsets)
        entity_heads_logits, entity_tails_logits = self.entity_model.predict([token_ids, segment_ids])
        entity_heads, entity_tails = np.where(entity_heads_logits[0] > threshold), np.where(entity_tails_logits[0] > threshold)
        subjects = []
        entity_map = {}
        for head, head_type in zip(*entity_heads):
            for tail, tail_type in zip(*entity_tails):
                if head <= tail and head_type == tail_type:
                    entity = self.decode_entity(text, mapping, head, tail)
                    if head_type == 0:
                        subjects.append((entity, head, tail))
                    else:
                        entity_map[head] = entity
                    break

        triple_set = set()
        if subjects:
            # translating decoding
            relations_logits = self.rel_model.predict([token_ids, segment_ids])
            relations = np.where(relations_logits[0] > threshold)[0].tolist()
            if relations:
                batch_sub_heads = []
                batch_sub_tails = []
                batch_rels = []
                batch_sub_entities = []
                batch_rel_types = []
                for (sub, sub_head, sub_tail) in subjects:
                    for rel in relations:
                        batch_sub_heads.append([sub_head])
                        batch_sub_tails.append([sub_tail])
                        batch_rels.append([rel])
                        batch_sub_entities.append(sub)
                        batch_rel_types.append(self.id2rel[rel])

                batch_token_ids = np.repeat(token_ids, len(subjects) * len(relations), 0)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                obj_head_logits = self.translate_model.predict_on_batch([
                    batch_token_ids, batch_segment_ids, np.array(batch_sub_heads), np.array(batch_sub_tails), np.array(batch_rels)
                ])
                for sub, rel, obj_head_logit in zip(batch_sub_entities, batch_rel_types, obj_head_logits):
                    for h in np.where(obj_head_logit > threshold)[0].tolist():
                        if h in entity_map:
                            obj = entity_map[h]
                            triple_set.add((sub, rel, obj))
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
