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

    def __init__(self, vocab_size, embeddig_dim, hidden_size, weight_tying=True, model_name=None):
        super(ClassicLanguageModel, self).__init__()
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.embeddig_dim = embeddig_dim
        self.hidden_size = hidden_size
        self.weight_tying = weight_tying
        
        self.embedding = nn.Embedding(vocab_size, embeddig_dim)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

        if weight_tying:
            if hidden_size != embeddig_dim:
                raise Exception("hidden_size and embeddig_dim mismatch")
            else:
                self.head.weight = self.embedding.weight

    def forward(self, batch):
        x = batch['text']
        pad_mask = batch['loss_mask']

        emb = self.embedding(x)
        emb = self.do(emb)
        pad_len = emb.shape[1]
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, 
            pad_mask.sum(dim=1),
            batch_first=True,
            enforce_sorted=False
        )
        lstm_output = self.lstm(emb)[0]
        gru_output = self.gru(lstm_output)[0]
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

        lstm_output = self.do(lstm_output)
        gru_output = self.do(gru_output)
        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)
        return F.log_softmax(self.head(pre_head), dim=2)



class AttentionLanguageModel(nn.Module):

    def __init__(self, vocab_size, embeddig_dim, hidden_size,  n_heads=8, weight_tying=True, model_name=None):
        super(AttentionLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.embeddig_dim = embeddig_dim
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.weight_tying = weight_tying
        
        self.embedding = nn.Embedding(vocab_size, embeddig_dim)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

        self.attention = nn.MultiheadAttention(hidden_size, n_heads)

        if weight_tying:
            if hidden_size != embeddig_dim:
                raise Exception("hidden_size and embeddig_dim mismatch")
            else:
                self.head.weight = self.embedding.weight

    def generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, batch):
        pad_mask = batch['loss_mask']
        x = batch['text']
        attn_mask = self.generate_square_subsequent_mask(x.shape[1])
        attn_mask = attn_mask.type_as(x)
        emb = self.embedding(x)
        emb = self.do(emb)
        pad_len = emb.shape[1]
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, 
            pad_mask.sum(dim=1),
            batch_first=True,
            enforce_sorted=False
        )
        lstm_output = self.lstm(emb)[0]
        gru_output = self.gru(lstm_output)[0]
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

        lstm_output = self.do(lstm_output)
        gru_output = self.do(gru_output)

        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)

        pad_mask = pad_mask.unsqueeze(2).repeat(1, 1, self.hidden_size)
        pre_head = pre_head * pad_mask

        pre_head = pre_head.transpose(0, 1)
        attn_output, _ = self.attention(pre_head, pre_head, pre_head, attn_mask=attn_mask)

        return F.log_softmax(self.head(attn_output), dim=2)

