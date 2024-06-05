# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from utils import compute_metrics

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output + x  # residual connection

class EntityModel(nn.Module):
    def __init__(self, bert_model_name):
        super(EntityModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.pred_entity_heads = nn.Linear(self.bert.config.hidden_size, 2)
        self.pred_entity_tails = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        token_type_ids = token_type_ids.to(self.bert.device)
        
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        tokens_feature = bert_outputs.last_hidden_state
        pred_entity_heads = torch.sigmoid(self.pred_entity_heads(tokens_feature))
        pred_entity_tails = torch.sigmoid(self.pred_entity_tails(tokens_feature))
        return pred_entity_heads, pred_entity_tails

class RelationModel(nn.Module):
    def __init__(self, bert_model_name, relation_size):
        super(RelationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.pred_rels = nn.Linear(self.bert.config.hidden_size, relation_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        token_type_ids = token_type_ids.to(self.bert.device)
        
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        tokens_feature = bert_outputs.last_hidden_state
        pred_rels = torch.sigmoid(self.pred_rels(tokens_feature[:, 0, :]))
        return pred_rels

class TranslateModel(nn.Module):
    def __init__(self, bert_model_name, relation_size, hidden_size):
        super(TranslateModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.rel_embedding = nn.Embedding(relation_size, hidden_size)
        self.rel_dense = nn.Linear(hidden_size, hidden_size)
        self.self_attention = SelfAttention(hidden_size)
        self.pred_obj_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, sub_head, sub_tail, rel):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        token_type_ids = token_type_ids.to(self.bert.device)
        sub_head = sub_head.to(self.bert.device)
        sub_tail = sub_tail.to(self.bert.device)
        rel = rel.to(self.bert.device)
        
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        tokens_feature = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    
        sub_head_feature = tokens_feature[torch.arange(tokens_feature.size(0)), sub_head.squeeze()]  # (batch_size, hidden_size)
        sub_tail_feature = tokens_feature[torch.arange(tokens_feature.size(0)), sub_tail.squeeze()]  # (batch_size, hidden_size)
        sub_feature = (sub_head_feature + sub_tail_feature) / 2  # (batch_size, hidden_size)
    
        rel_feature = torch.relu(self.rel_dense(self.rel_embedding(rel).squeeze(1)))  # (batch_size, hidden_size)
    
        # Ensure dimensions match for broadcasting
        sub_feature = sub_feature.unsqueeze(1).expand(-1, tokens_feature.size(1), -1)  # (batch_size, seq_len, hidden_size)
        rel_feature = rel_feature.unsqueeze(1).expand(-1, tokens_feature.size(1), -1)  # (batch_size, seq_len, hidden_size)
    
        obj_feature = tokens_feature + sub_feature + rel_feature  # (batch_size, seq_len, hidden_size)
        obj_feature = self.self_attention(obj_feature)  # (batch_size, seq_len, hidden_size)
        pred_obj_head = torch.sigmoid(self.pred_obj_head(obj_feature))  # (batch_size, seq_len, 1)
    
        return pred_obj_head.squeeze(-1)  # (batch_size, seq_len)

class TDEERModel(nn.Module):
    def __init__(self, entity_model, rel_model, translate_model):
        super(TDEERModel, self).__init__()
        self.entity_model = entity_model
        self.rel_model = rel_model
        self.translate_model = translate_model

    def forward(self, input_ids, attention_mask, token_type_ids, gold_entity_heads, gold_entity_tails, gold_rels, sub_head, sub_tail, rel, gold_obj_head):
        input_ids = input_ids.to(self.entity_model.bert.device)
        attention_mask = attention_mask.to(self.entity_model.bert.device)
        token_type_ids = token_type_ids.to(self.entity_model.bert.device)
        gold_entity_heads = gold_entity_heads.to(self.entity_model.bert.device)
        gold_entity_tails = gold_entity_tails.to(self.entity_model.bert.device)
        gold_rels = gold_rels.to(self.entity_model.bert.device)
        sub_head = sub_head.to(self.entity_model.bert.device)
        sub_tail = sub_tail.to(self.entity_model.bert.device)
        rel = rel.to(self.entity_model.bert.device)
        gold_obj_head = gold_obj_head.to(self.entity_model.bert.device)
        
        pred_entity_heads, pred_entity_tails = self.entity_model(input_ids, attention_mask, token_type_ids)
        pred_rels = self.rel_model(input_ids, attention_mask, token_type_ids)
        pred_obj_head = self.translate_model(input_ids, attention_mask, token_type_ids, sub_head, sub_tail, rel)

        loss = self.compute_loss(pred_entity_heads, gold_entity_heads, pred_entity_tails, gold_entity_tails, pred_rels, gold_rels, pred_obj_head, gold_obj_head)
        return loss

    def compute_loss(self, pred_entity_heads, gold_entity_heads, pred_entity_tails, gold_entity_tails, pred_rels, gold_rels, pred_obj_head, gold_obj_head):
        entity_heads_loss = nn.functional.binary_cross_entropy(pred_entity_heads, gold_entity_heads, reduction='mean')
        entity_tails_loss = nn.functional.binary_cross_entropy(pred_entity_tails, gold_entity_tails, reduction='mean')
        rel_loss = nn.functional.binary_cross_entropy(pred_rels, gold_rels, reduction='mean')
        pred_obj_head = pred_obj_head.squeeze(-1)
        obj_head_loss = nn.functional.binary_cross_entropy(pred_obj_head, gold_obj_head, reduction='mean')
    
        total_loss = entity_heads_loss + entity_tails_loss + rel_loss + 5.0 * obj_head_loss
        return total_loss

def build_model(bert_model_name: str, learning_rate: float, relation_size: int, device):
    hidden_size = BertModel.from_pretrained(bert_model_name).config.hidden_size
    entity_model = EntityModel(bert_model_name).to(device)
    rel_model = RelationModel(bert_model_name, relation_size).to(device)
    translate_model = TranslateModel(bert_model_name, relation_size, hidden_size).to(device)
    train_model = TDEERModel(entity_model, rel_model, translate_model).to(device)

    optimizer = optim.AdamW(train_model.parameters(), lr=learning_rate)
    return entity_model, rel_model, translate_model, train_model, optimizer

class Evaluator:
    def __init__(self, infer, model, dev_data, save_weights_path, model_name, optimizer, device, learning_rate=5e-5, min_learning_rate=5e-6):
        self.infer = infer
        self.model = model
        self.dev_data = dev_data
        self.save_weights_path = save_weights_path
        self.model_name = model_name
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.best = float('-inf')
        self.passed = 0
        self.is_exact_match = model_name.startswith('NYT11-HRL')

    def evaluate(self):
        self.model.eval()
        precision, recall, f1 = compute_metrics(self.infer, self.dev_data, exact_match=self.is_exact_match, model_name=self.model_name)
        if f1 > self.best:
            self.best = f1
            torch.save(self.model.state_dict(), self.save_weights_path)
            print("New best result!")
        print(f'f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, best f1: {self.best:.4f}')
        self.model.train()

    def adjust_learning_rate(self, step, total_steps):
        if step < total_steps:
            lr = (step + 1) / total_steps * self.learning_rate
        else:
            lr = (2 - (step + 1) / total_steps) * (self.learning_rate - self.min_learning_rate) + self.min_learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Example of usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entity_model, rel_model, translate_model, train_model, optimizer = build_model('bert-base-uncased', 2e-5, 10, device)
evaluator = Evaluator(None, train_model, None, 'model.pth', 'NYT11-HRL', optimizer, device)
