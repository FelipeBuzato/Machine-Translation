import math
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
        
        if(model_name == "Transformer"):
            return Transformer(**params)
        
        raise ValueError(f"Unknown model: {model_name}")


class Transformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_target, embedding_dim=128, n_heads=8, num_encoder_layers=2, num_decoder_layers=2,
        dim_feed_forward=1024, dropout=0.1, padding_idx=0, BOS_index=2, EOS_index=3, max_length_decoded=40):
        super().__init__()

        self.padding_idx = padding_idx
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index
        self.vocab_size_target = vocab_size_target
        self.max_length_decoded = max_length_decoded

        ## Embeddings
        self.src_embedding = nn.Embedding(vocab_size_src, embedding_dim, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(vocab_size_target, embedding_dim, padding_idx=padding_idx)

        ## Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim==embedding_dim, dropout=dropout)

        ## Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        ## Transformer Decoder 
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size_target)


    def forward(self, X, y):

        # src: (batch, src_len)
        # tgt: (batch, tgt_len)

        tgt_input = y[:, :-1]

        # Masks
        src_padding_mask = (X == self.padding_idx)
        tgt_padding_mask = (tgt_input == self.padding_idx)
        tgt_mask = torch.triu(torch.ones(tgt_input.size(1), tgt_input.size(1), dtype=torch.bool, device=X.device), diagonal=1)

        # Src Embedding
        X = self.src_embedding(X)

        # Src positional encoding
        X = self.positional_encoding(X)

        # encoder
        memory = self.encoder(X, src_key_padding_mask=src_padding_mask)

        # Target Embedding
        tgt_input = self.tgt_embedding(tgt_input)

        # Target positional encoding
        tgt_input = self.positional_encoding(tgt_input)

        # decoder
        output = self.decoder(
            tgt=tgt_input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        # output probability logits
        logits = self.fc(output)

        # the following opperation is done with the sole purppose of
        # shaping the output so that the trainer will compute the loss correctly
        zeros = torch.zeros(logits.size(0), 1, logits.size(2), device=logits.device, dtype=logits.dtype)
        logits = torch.cat([zeros, logits], dim=1)

        return logits
    

    @torch.no_grad()
    def predict(self, X):

        # Src mask
        src_padding_mask = (X == self.padding_idx)

        # Src embedding
        X_emb = self.src_embedding(X)

        # Src positional encoding
        X_emb = self.positional_encoding(X_emb)

        # Encoder
        memory = self.encoder(X_emb, src_key_padding_mask=src_padding_mask)

        # Output fixo
        batch_size = X.size(0)
        generated = torch.full((batch_size, self.max_length_decoded + 1), self.padding_idx, dtype=torch.long, device=X.device)
        # Primeiro token é BOS
        generated[:, 0] = self.BOS_index

        finished = torch.zeros(batch_size, dtype=torch.bool, device=X.device)

        for t in range(self.max_length_decoded):

            # Apenas tokens já gerados até agora
            tgt = generated[:, :t+1]

            # Target masks
            tgt_padding_mask = (tgt == self.padding_idx)
            tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=X.device), diagonal=1)

            # Target embedding
            tgt_emb = self.tgt_embedding(tgt)

            # Positional encoding
            tgt_emb = self.positional_encoding(tgt_emb)

            # Decoder
            output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            # Logits apenas do último token
            logits = self.fc(output[:, -1])
            next_token = logits.argmax(dim=1)

            # Não altera sequências que já terminaram
            next_token = torch.where(finished, torch.tensor(self.EOS_index, device=X.device), next_token)
            generated[:, t+1] = next_token

            finished |= (next_token == self.EOS_index)

            if finished.all():
                break

        return generated[:, 1:]
    

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # (max_len, d_model)
        pe = torch.zeros(max_len, embedding_dim)

        # (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1).float()

        # Frequências das senóides
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Não é um parâmetro treinável, mas acompanha o dispositivo do modelo
        self.register_buffer("pe", pe)


    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

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