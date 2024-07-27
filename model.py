import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class SelfAttention(nn.Module):
    def __init__(self, input_size, is_residual=True, attention_activation='relu'):
        super(SelfAttention, self).__init__()
        self.is_residual = is_residual
        self.attention = nn.Linear(input_size, 1)
        self.activation = nn.ReLU() if attention_activation == 'relu' else nn.Tanh()

    def forward(self, inputs):
        attention_weights = self.activation(self.attention(inputs))
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(inputs * attention_weights, dim=1)
        if self.is_residual:
            outputs = inputs + attended.unsqueeze(1)
        else:
            outputs = attended.unsqueeze(1)
        return outputs
        
class TDEER(nn.Module):
    def __init__(self, bert_model_name, relation_size):
        super(TDEER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.relation_size = relation_size
        
        hidden_size = self.bert.config.hidden_size
        
        self.entity_heads = nn.Linear(hidden_size, 2)
        self.entity_tails = nn.Linear(hidden_size, 2)
        self.rel_classifier = nn.Linear(hidden_size, relation_size)
        
        self.sub_head_feature = nn.Linear(hidden_size, hidden_size)
        self.sub_tail_feature = nn.Linear(hidden_size, hidden_size)
        self.rel_feature = nn.Embedding(relation_size, hidden_size)
        self.rel_feature_dense = nn.Linear(hidden_size, hidden_size)
        
        # MODIFIED: Enhanced obj_feature
        self.obj_feature = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.self_attention = nn.Linear(hidden_size, 1)
        self.pred_obj_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, entity_heads=None, entity_tails=None, rels=None, 
                sample_subj_head=None, sample_subj_tail=None, sample_rel=None, sample_obj_heads=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
    
        pred_entity_heads = self.entity_heads(sequence_output)
        pred_entity_tails = self.entity_tails(sequence_output)
        pred_rels = self.rel_classifier(pooled_output)
    
        if sample_subj_head is not None and sample_subj_tail is not None and sample_rel is not None:
            batch_size, seq_len = input_ids.size()
            
            sub_head_feature = self.sub_head_feature(sequence_output[torch.arange(batch_size), sample_subj_head])
            sub_tail_feature = self.sub_tail_feature(sequence_output[torch.arange(batch_size), sample_subj_tail])
            sub_feature = (sub_head_feature + sub_tail_feature) / 2
    
            rel_feature = self.rel_feature(sample_rel)
            rel_feature = F.relu(self.rel_feature_dense(rel_feature))
    
            obj_feature_input = torch.cat([sequence_output, 
                                           sub_feature.unsqueeze(1).expand(-1, seq_len, -1), 
                                           rel_feature.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
            obj_feature = self.obj_feature(obj_feature_input)
    
            # attention_weights = F.softmax(self.self_attention(obj_feature), dim=1)
            # attended_obj_feature = (obj_feature * attention_weights).sum(dim=1)
            
            # 修改这里
            pred_obj_head = self.pred_obj_head(obj_feature).squeeze(-1)
    
            # 添加调试信息
            # print(f"Debug - pred_obj_head shape: {pred_obj_head.shape}")
            # print(f"Debug - attention_mask shape: {attention_mask.shape}")
    
            return pred_entity_heads, pred_entity_tails, pred_rels, pred_obj_head
    
        return pred_entity_heads, pred_entity_tails, pred_rels

class Evaluator:
    def __init__(self, model, dev_data, tokenizer, id2rel, device):
        self.model = model
        self.dev_data = dev_data
        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.device = device
        self.best_f1 = float('-inf')

    def evaluate(self, epoch):
        self.model.eval()
        # Implement evaluation logic here
        # Use the compute_metrics function from utils.py
        precision, recall, f1 = compute_metrics(self.model, self.dev_data, self.tokenizer, self.id2rel, self.device)
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(self.model.state_dict(), f'best_model_epoch_{epoch}.pt')
            print("New best model saved!")
        
        print(f'Epoch {epoch}: F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Best F1: {self.best_f1:.4f}\n')
        
        return f1

def build_model(bert_dir: str, relation_size: int, device: str):
    model = TDEER(bert_dir, relation_size).to(device)
    return model

def get_optimizer(model: nn.Module, learning_rate: float):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)