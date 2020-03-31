import torch
import torch.nn as nn
import torch.nn.functional as F 



class SpatialDropout(torch.nn.Dropout2d):
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        
        return x



class ClassicLanguageModel(nn.Module):

    def __init__(self, vocab_size, embeddig_dim, hidden_size):
        super(ClassicLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeddig_dim)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch):
        emb = self.embedding(batch)
        emb = self.do(emb)
        lstm_output = self.do(self.lstm(emb)[0])
        gru_output = self.do(self.gru(lstm_output)[0])
        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)
        return F.log_softmax(self.head(pre_head), dim=2)



class AttentionLanguageModel(nn.Module):

    def __init__(self, vocab_size, embeddig_dim, hidden_size,  n_heads=8):
        super(AttentionLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeddig_dim)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

        self.attention = nn.MultiheadAttention(hidden_size, n_heads)

    def generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, batch):
        emb = self.embedding(batch)
        emb = self.do(emb)
        lstm_output = self.do(self.lstm(emb)[0])
        gru_output = self.do(self.gru(lstm_output)[0])
        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)

        pre_head = pre_head.transpose(0, 1)
        mask = self.generate_square_subsequent_mask(pre_head.shape[0])
        attn_output, _ = self.attention(pre_head, pre_head, pre_head, attn_mask=mask)

        return F.log_softmax(self.head(attn_output), dim=2)

