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
        self.emb_do = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embeddig_dim, hidden_size, batch_first=True)
        self.do1 = SpatialDropout()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.do2 = SpatialDropout()
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch):
        emb = self.embedding(batch)
        emb = self.emb_do(emb)
        lstm_output = self.do1(self.lstm(emb)[0])
        gru_output = self.do2(self.gru(lstm_output)[0])
        pre_head = lstm_output + gru_output
        pre_head = self.norm(pre_head)
        return F.log_softmax(self.head(pre_head), dim=2)

