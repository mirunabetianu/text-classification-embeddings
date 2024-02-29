import torch
import torch.nn as nn

# Simple LSTM Model 
# Linear layer for mapping the size of the hidden layer to the number of classes
# The embedding layer is skipped since the embeddings are given
class LSTMModel(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out