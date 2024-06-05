import json
import time
from typing import Dict, List, Set

import torch
import numpy as np
from tqdm import tqdm


def rematch(offsets: List) -> List:
    mapping = []
    for offset in offsets:
        if offset[0] == 0 and offset[1] == 0:
            mapping.append([])
        else:
            mapping.append(list(range(offset[0], offset[1])))
    return mapping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Infer:
    def __init__(self, entity_model: torch.nn.Module, rel_model: torch.nn.Module, translate_model: torch.nn.Module,
                 tokenizer: object, id2rel: Dict):
        self.entity_model = entity_model.to(device)
        self.rel_model = rel_model.to(device)
        self.translate_model = translate_model.to(device)
        self.tokenizer = tokenizer
        self.id2rel = id2rel

    def decode_entity(self, text: str, mapping: List, start: int, end: int):
        # print(f"Input text: {text}")
        # print(f"Mapping: {mapping}")
        # print(f"Start index: {start}, End index: {end}")
        
        if start >= len(mapping) or end >= len(mapping):
            print("Start or end index out of range.")
            return ""
        
        s = mapping[start] if start < len(mapping) else []
        e = mapping[end] if end < len(mapping) else []
        # print(f"Raw start mapping: {s}, Raw end mapping: {e}")
        
        # 取s和e的第一个和最后一个值来确保提取正确的子串
        s = s[0] if s else 0
        e = e[-1] if e else len(text) - 1
        # print(f"Adjusted start: {s}, Adjusted end: {e}")
        
        s = max(0, s)
        e = min(len(text) - 1, e)
        
        entity = text[s: e + 1]
        # print(f"Extracted entity: {entity}")
        return entity


    def __call__(self, text: str, threshold: float = 0.5) -> Set:
        # Tokenize text and prepare input tensors
        tokened = self.tokenizer.encode_plus(text, add_special_tokens=True, return_offsets_mapping=True) # 确保添加了特殊标记
        token_ids = torch.tensor([tokened.input_ids], dtype=torch.long).to(device)
        segment_ids = torch.tensor([tokened.token_type_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([tokened.attention_mask], dtype=torch.long).to(device)
        mapping = rematch(tokened.offset_mapping)
        
        # Run entity model to get logits for entity heads and tails
        with torch.no_grad():
            entity_heads_logits, entity_tails_logits = self.entity_model(token_ids, attention_mask, segment_ids)
        
        # Apply threshold to get entity indices
        entity_heads = (entity_heads_logits > threshold).nonzero(as_tuple=True)
        entity_tails = (entity_tails_logits > threshold).nonzero(as_tuple=True)
        subjects = []
        entity_map = {}
        
        # print(f"entity_heads_logits shape: {entity_heads_logits.shape}")
        # print(f"entity_tails_logits shape: {entity_tails_logits.shape}")
        # print(f"entity_heads: {entity_heads}")
        # print(f"entity_tails: {entity_tails}")
        
        # Generate potential subjects and entities from heads and tails
        for head_idx, head_type_idx in zip(entity_heads[1], entity_heads[2]):
            for tail_idx, tail_type_idx in zip(entity_tails[1], entity_tails[2]):
                if head_idx <= tail_idx and head_type_idx == tail_type_idx:
                    entity = self.decode_entity(text, mapping, head_idx.item(), tail_idx.item())
                    # print(f"Extracted entity: {entity} from indices {head_idx.item()} to {tail_idx.item()}")
                    if head_type_idx == 0:  # Assuming type 0 are subjects
                        subjects.append((entity, head_idx.item(), tail_idx.item()))
                    else:
                        entity_map[head_idx.item()] = entity
                        
        print(f"Subjects: {subjects}")
        print(f"Entity map: {entity_map}")

        triple_set = set()
        if subjects:
            with torch.no_grad():
                relations_logits = self.rel_model(token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
            relations = (relations_logits > threshold).nonzero(as_tuple=True)
            if relations[0].numel() > 0:
                batch_size = len(subjects)
                for sub, sub_head, sub_tail in subjects:
                    for rel_idx in relations[0]:
                        rel = self.id2rel[rel_idx.item()]
                        batch_token_ids = token_ids.expand(batch_size, -1)
                        batch_segment_ids = segment_ids.expand(batch_size, -1)
                        batch_attention_mask = attention_mask.expand(batch_size, -1)
                        batch_subj_head = torch.tensor([sub_head], dtype=torch.long).expand(batch_size, 1).to(device)
                        batch_subj_tail = torch.tensor([sub_tail], dtype=torch.long).expand(batch_size, 1).to(device)
                        batch_rels = torch.tensor([rel_idx.item()], dtype=torch.long).expand(batch_size, 1).to(device)
    
                        # Debugging shapes
                        # print(f"batch_token_ids shape: {batch_token_ids.shape}")
                        # print(f"batch_segment_ids shape: {batch_segment_ids.shape}")
                        # print(f"batch_attention_mask shape: {batch_attention_mask.shape}")
                        # print(f"batch_subj_head shape: {batch_subj_head.shape}")
                        # print(f"batch_subj_tail shape: {batch_subj_tail.shape}")
                        # print(f"batch_rels shape: {batch_rels.shape}")
    
                        obj_head_logits = self.translate_model(batch_token_ids, batch_attention_mask, batch_segment_ids, batch_subj_head, batch_subj_tail, batch_rels)
                        for obj_head_idx in obj_head_logits.argmax(dim=1).tolist():
                            if obj_head_idx in entity_map:
                                obj = entity_map[obj_head_idx]
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for line in tqdm(iter(dev_data)):
        start_time = time.time()
        pred_triples = infer(line['text'])
        infer_times.append(time.time() - start_time)
        gold_triples = set(line['triple_list'])
        # print(f"Predicted triples: {pred_triples}")
        # print(f"Gold triples: {gold_triples}")

        if exact_match:
            gold_triples = remove_space(gold_triples)
            pred_triples = remove_space(pred_triples)

        pred_triples_eval, gold_triples_eval = partial_match(pred_triples, gold_triples) if not exact_match else (pred_triples, gold_triples)

        print(f"Predicted triples (eval): {pred_triples_eval}")
        print(f"Gold triples (eval): {gold_triples_eval}")
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

