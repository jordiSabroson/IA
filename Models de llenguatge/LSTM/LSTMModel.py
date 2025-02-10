from torch import nn
class CustomLSTMModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_size=100, hidden_size=128):
        super(CustomLSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.h0 = torch.zeros(1,1,hidden_size)
        # self.c0 = torch.zeros(1,1,hidden_size)
        # self.last_layer = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(embed_size, 2)
        # )
    
    def forward(self, x):
        embedded = self.embed(x)
        lstm_out, _ = self.lstm(embedded)
        # out_encod, (hn, cn) = self.encoder(emb, (h0, c0))
        logits = self.fc(lstm_out[:, -1])
        return logits