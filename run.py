# -*- coding:utf-8 -*-

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from dataloader import DataGenerator, load_data, load_rel, collate_fn
from model import build_model, Evaluator
from utils import Infer, compute_metrics

parser = argparse.ArgumentParser(description='TDEER cli')
parser.add_argument('--do_train', action='store_true', help='to train TDEER, plz specify --do_train')
parser.add_argument('--do_test', action='store_true', help='specify --do_test to evaluate')
parser.add_argument('--model_name', type=str, required=True, help='specify the model name')
parser.add_argument('--rel_path', type=str, required=True, help='specify the relation path')
parser.add_argument('--train_path', type=str, help='specify the train path')
parser.add_argument('--dev_path', type=str, help='specify the dev path')
parser.add_argument('--test_path', type=str, help='specify the test path')
parser.add_argument('--bert_model_name', type=str, default='bert-base-cased', help='specify the pre-trained bert model')
parser.add_argument('--save_path', default=None, type=str, help='specify the save path to save model [training phase]')
parser.add_argument('--ckpt_path', default=None, type=str, help='specify the ckpt path [test phase]')
parser.add_argument('--learning_rate', default=2e-5, type=float, help='specify the learning rate')
parser.add_argument('--epoch', default=100, type=int, help='specify the epoch size')
parser.add_argument('--batch_size', default=8, type=int, help='specify the batch size')
parser.add_argument('--max_len', default=120, type=int, help='specify the max len')
parser.add_argument('--neg_samples', default=None, type=int, help='specify negative sample num')
parser.add_argument('--max_sample_triples', default=None, type=int, help='specify max sample triples')
parser.add_argument('--verbose', default=2, type=int, help='specify verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch')
args = parser.parse_args()

print("Argument:", args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2rel, rel2id, all_rels = load_rel(args.rel_path)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
entity_model, rel_model, translate_model, train_model, optimizer = build_model(args.bert_model_name, args.learning_rate, len(all_rels), device)
train_model.to(device)

if args.do_train:
    assert args.save_path is not None, "please specify --save_path in training phase"
    
    # 加载训练数据、验证数据和测试数据
    train_data = load_data(args.train_path, rel2id, is_train=True)
    dev_data = load_data(args.dev_path, rel2id, is_train=False)
    test_data = load_data(args.test_path, rel2id, is_train=False) if args.test_path is not None else None
    
    # 创建数据生成器和数据加载器
    train_generator = DataGenerator(train_data, tokenizer, rel2id, all_rels, args.max_len, args.max_sample_triples, args.neg_samples)
    train_loader = DataLoader(train_generator, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 创建优化器和评估器
    # optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)
    infer = Infer(entity_model, rel_model, translate_model, tokenizer, id2rel)
    evaluator = Evaluator(infer, train_model, dev_data, args.save_path, args.model_name, optimizer, device)
    
    # 训练模型
    for epoch in range(args.epoch):
        train_model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch}', unit='batch') as t_loader:
            for batch in t_loader:
                batch = {k: v.to(device).float() if v.dtype == torch.float64 else v.to(device) for k, v in batch.items()}
                # print(batch)
                inputs = {
                    'input_ids': batch['token_ids'],  # Changed from 'input_ids' to 'token_ids'
                    'attention_mask': batch['attention_mask'],  # Assuming this key is correct
                    'token_type_ids': batch['segment_ids'],  # Changed from 'token_type_ids' to 'segment_ids'
                    'gold_entity_heads': batch['entity_heads'],  # Assuming this key is correct
                    'gold_entity_tails': batch['entity_tails'],  # Assuming this key is correct
                    'gold_rels': batch['rels'],  # Assuming this key is correct
                    'sub_head': batch['sample_subj_head'],  # Assuming this key is correct
                    'sub_tail': batch['sample_subj_tail'],  # Assuming this key is correct
                    'rel': batch['sample_rel'],  # Assuming this key is correct
                    'gold_obj_head': batch['sample_obj_heads']  # Assuming this key is correct
                }


                optimizer.zero_grad()
                loss = train_model(**inputs)
                loss.backward()
                optimizer.step()
                t_loader.set_postfix({'loss': loss.item()})
                t_loader.update()
        
        # 每个 epoch 结束后进行评估
        evaluator.evaluate()

    # 保存训练好的模型
    torch.save(train_model.state_dict(), args.save_path)

if args.do_test:
    assert args.ckpt_path is not None, "please specify --ckpt_path in test phase"
    
    # 加载测试数据和模型
    test_data = load_data(args.test_path, rel2id, is_train=False)
    train_model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    train_model.to(device)
    train_model.eval()
    
    # 进行测试并计算评估指标
    infer = Infer(entity_model, rel_model, translate_model, tokenizer, id2rel)
    precision, recall, f1_score = compute_metrics(infer, test_data, model_name=args.model_name)
    print(f'precision: {precision}, recall: {recall}, f1: {f1_score}')
