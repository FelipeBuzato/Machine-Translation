import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params=None):
        if params is None:
            params = {}
        
        if(model_name == "LSTM"):
            return LSTM(**params)
        
        raise ValueError(f"Unknown model: {model_name}")
    

class LSTM(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_target, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.3, 
                 padding_idx=0, BOS_index=2, EOS_index=3, teacher_forcing=0.5, max_length_decoded=100):
        super().__init__()

        self.encoder_lstm = EncoderLSTM(vocab_size_src, embedding_dim, hidden_dim, num_layers, dropout, padding_idx)
        self.decoder_lstm = DecoderLSTM(vocab_size_target, embedding_dim, hidden_dim, num_layers, dropout, padding_idx)
        self.vocab_size_target = vocab_size_target
        self.teacher_forcing = teacher_forcing
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index
        self.max_length_decoded = max_length_decoded


    def forward(self, X, y):

        batch_size = X.size(0)
        target_length = y.size(1)

        outputs = torch.zeros(batch_size, target_length, self.vocab_size_target, device=X.device)

        ## LSTM Encoder
        hidden, cell = self.encoder_lstm(X)
 
        ## LSTM Decoder
        # Primeiro input do decoder = <BOS>
        decoder_input = y[:, 0]

        use_teacher = torch.rand(target_length, device=X.device) < self.teacher_forcing

        for t in range(1, target_length):

            logits, hidden, cell = self.decoder_lstm(decoder_input, hidden, cell)
            outputs[:, t] = logits

            predicted = logits.argmax(dim=1)

            decoder_input = y[:, t] if use_teacher[t] else predicted

        return outputs
    

    @torch.no_grad()
    def predict(self, X):

        ## Encode
        hidden, cell = self.encoder_lstm(X)

        ## Decode
        batch_size = X.size(0)
        decoder_input = torch.full((batch_size,), self.BOS_index, dtype=torch.long, device=X.device)
        predictions = torch.zeros(batch_size, self.max_length_decoded, dtype=torch.long, device=X.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=X.device)

        for t in range(self.max_length_decoded):

            logits, hidden, cell = self.decoder_lstm(decoder_input, hidden, cell)
            predicted = logits.argmax(dim=1)
            predictions[:, t] = predicted

            finished |= predicted == self.EOS_index
            if finished.all():
                break

            decoder_input = predicted

        return predictions
    

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )


    def forward(self, X):
        lengths = (X != 0).sum(dim=1).cpu()
        embedded = self.embedding(X)

        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, cell) = self.lstm(packed)

        return hidden, cell
    

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, token, hidden, cell):
        # token: (batch,)

        embedded = self.embedding(token)  # (batch, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # output: (batch, 1, hidden_dim)
        output = output.squeeze(1)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, vocab_size)

        return logits, hidden, cell