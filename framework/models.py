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
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embeddig_dim)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

        self.attention = nn.MultiheadAttention(hidden_size, n_heads)


    def forward(self, batch):
        pad_mask = batch[2]
        attn_mask = batch[1]
        batch = batch[0]
        emb = self.embedding(batch)
        emb = self.do(emb)
        pad_len = emb.shape[1]
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, 
            pad_mask.sum(dim=1),
            batch_first=True,
            enforce_sorted=False
        )
        lstm_output = self.do(self.lstm(emb)[0])
        gru_output = self.do(self.gru(lstm_output)[0])
        lstm_output = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, 
            batch_first=True, 
            total_length=pad_len
        )[0]
        gru_output = torch.nn.utils.rnn.pad_packed_sequence(
            gru_output, 
            batch_first=True, 
            total_length=pad_len
        )[0]

        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)

        pad_mask = pad_mask.unsqueeze(2).repeat(1, 1, self.hidden_size)
        pre_head = pre_head * pad_mask

        pre_head = pre_head.transpose(0, 1)
        attn_output, _ = self.attention(pre_head, pre_head, pre_head, attn_mask=attn_mask)

        return F.log_softmax(self.head(attn_output), dim=2)

