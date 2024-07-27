import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from transformers import BertTokenizerFast
from tqdm import tqdm
import numpy as np

from dataloader import TDEERDataset, load_data, load_rel
from model import TDEER, get_optimizer
from utils import Infer, compute_metrics

parser = argparse.ArgumentParser(description='TDEER cli')
parser.add_argument('--do_train', action='store_true', help='to train TDEER, plz specify --do_train')
parser.add_argument('--do_test', action='store_true', help='specify --do_test to evaluate')
parser.add_argument('--model_name', type=str, required=True, help='specify the model name')
parser.add_argument('--rel_path', type=str, required=True, help='specify the relation path')
parser.add_argument('--train_path', type=str, help='specify the train path')
parser.add_argument('--dev_path', type=str, help='specify the dev path')
parser.add_argument('--test_path', type=str, help='specify the test path')
parser.add_argument('--bert_model', type=str, default='bert-base-cased', help='specify the pre-trained bert model')
parser.add_argument('--save_path', default=None, type=str, help='specify the save path to save model [training phase]')
parser.add_argument('--ckpt_path', default=None, type=str, help='specify the ckpt path [test phase]')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='specify the learning rate')
parser.add_argument('--epoch', default=5, type=int, help='specify the epoch size')
parser.add_argument('--batch_size', default=16, type=int, help='specify the batch size')
parser.add_argument('--max_len', default=120, type=int, help='specify the max len')
parser.add_argument('--neg_samples', default=2, type=int, help='specify negative sample num')
parser.add_argument('--max_sample_triples', default=100, type=int, help='specify max sample triples')
parser.add_argument('--eval_steps', default=100, type=int, help='evaluate every N steps')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
parser.add_argument('--subset_size', type=int, default=None, help='use a subset of data for quick validation')

args = parser.parse_args()

print("Argument:", args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id2rel, rel2id, all_rels = load_rel(args.rel_path)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)
model = TDEER(args.bert_model, len(all_rels)).to(device)

def compute_loss(pred_entity_heads, pred_entity_tails, pred_rels, pred_obj_head,
                 entity_heads, entity_tails, rels, sample_obj_heads, attention_mask):
    entity_heads_loss = F.binary_cross_entropy_with_logits(pred_entity_heads, entity_heads, reduction='none')
    entity_heads_loss = (entity_heads_loss * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
    
    entity_tails_loss = F.binary_cross_entropy_with_logits(pred_entity_tails, entity_tails, reduction='none')
    entity_tails_loss = (entity_tails_loss * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
    
    rel_loss = F.binary_cross_entropy_with_logits(pred_rels, rels)
    
    obj_head_loss = F.binary_cross_entropy_with_logits(pred_obj_head, sample_obj_heads, reduction='none')
    obj_head_loss = (obj_head_loss * attention_mask).sum() / attention_mask.sum()
    
    loss = entity_heads_loss + entity_tails_loss + rel_loss + 10.0 * obj_head_loss
    return loss

if args.do_train:
    train_data = load_data(args.train_path, rel2id, is_train=True)
    dev_data = load_data(args.dev_path, rel2id, is_train=False)
    
    if args.subset_size:
        train_data = train_data[:args.subset_size]
        dev_data = dev_data[:min(args.subset_size, len(dev_data))]
    
    train_dataset = TDEERDataset(train_data, tokenizer, rel2id, all_rels, args.max_len, args.max_sample_triples, args.neg_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    optimizer = get_optimizer(model, args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    scaler = GradScaler()
    
    best_f1 = 0
    global_step = 0
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}") as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if 'token_ids' in batch:
                    batch['input_ids'] = batch.pop('token_ids')
                
                with autocast(enabled=args.use_amp):
                    outputs = model(**batch)
                    loss = compute_loss(*outputs, 
                                        batch['entity_heads'], 
                                        batch['entity_tails'], 
                                        batch['rels'], 
                                        batch['sample_obj_heads'], 
                                        batch['attention_mask'])
                
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

                global_step += 1
                if global_step % args.eval_steps == 0:
                    model.eval()
                    with torch.no_grad():
                        infer = Infer(model, tokenizer, id2rel, device)
                        precision, recall, f1 = compute_metrics(infer, dev_data, args.model_name)
                    print(f"Step {global_step} - Dev set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(model.state_dict(), args.save_path)
                        print(f"New best model saved with F1: {best_f1:.4f}")
                    model.train()

        print(f"Epoch {epoch+1}/{args.epoch}, Average Loss: {total_loss / len(train_dataloader):.4f}")
        scheduler.step()
        
        # Evaluate on dev set
        model.eval()
        with torch.no_grad():
            infer = Infer(model, tokenizer, id2rel, device)
            precision, recall, f1 = compute_metrics(infer, dev_data, args.model_name)
        print(f"Dev set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved with F1: {best_f1:.4f}")
            
if args.do_test:
    assert args.ckpt_path is not None, "please specify --ckpt_path in test phase"
    test_data = load_data(args.test_path, rel2id, is_train=False)
    model.load_state_dict(torch.load(args.ckpt_path))
    infer = Infer(model, tokenizer, id2rel, device)
    precision, recall, f1_score = compute_metrics(infer, test_data, args.model_name)
    print(f'Test set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}')