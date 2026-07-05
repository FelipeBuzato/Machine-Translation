import re
from collections import Counter
import numpy as np
import torch
import sentencepiece as spm
import tempfile
import os

class TokenizerFullWords:
    def __init__(self, min_freq=2, max_length=100):
        self.min_freq = min_freq
        self.max_length = max_length
        self.token_pattern = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]", flags=re.UNICODE)  
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        self.word2idx_, self.idx2word_ = None, None
        self.vocab_size_ = None


    def tokenize(self, text):
        # the tokenizer keeps hiffens inside words and punctuation characters become single characters
        return self.token_pattern.findall(text)
    

    def build_vocabulary(self, sentences):
        counter = Counter(token for sentence in sentences for token in sentence)
        vocab = list(token for token, freq in counter.most_common() if freq >= self.min_freq)
        word2idx = {word: idx for idx, word in enumerate(self.special_tokens + vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word
    

    def encode(self, X):
        X_indices = np.full((len(X), self.max_length+2), fill_value=self.word2idx_["<PAD>"], dtype=np.int64)
        for i, sentence in enumerate(X):
            indices = [self.word2idx_["<BOS>"]]
            indices += [
                self.word2idx_.get(word, self.word2idx_["<UNK>"])
                for word in sentence[:self.max_length]
            ]
            indices += [self.word2idx_["<EOS>"]]
            X_indices[i, :len(indices)] = indices

        return X_indices


    def decode(self, X):
        sentences = []

        for sentence in X.tolist():
            words = []
            for idx in sentence:
                if idx == 3:
                    break
                if idx in (0, 1, 2):
                    continue
                words.append(self.idx2word_[idx])
            sentences.append(" ".join(words))

        return sentences


    def fit(self, X, y=None):
        tokenized = X.iloc[:,0].apply(self.tokenize)

        # learn vocabulary
        self.word2idx_, self.idx2word_ = self.build_vocabulary(tokenized)
        self.vocab_size_ = len(self.word2idx_)
        print(f"vocab size: {self.vocab_size_}.")

        return self


    def transform(self, X):
        X = X.copy()

        # tokenize X
        X = X.iloc[:,0].apply(self.tokenize)

        lengths = [len(t) for t in X]

        print("Full Words")
        print(np.mean(lengths))
        print(np.percentile(lengths,[50,90,95,99]))

        # encode X (words -> index)
        X_encoded = self.encode(X)
        
        # transform the output into a tensor (n_samples, max_length)
        X_tensor = torch.from_numpy(X_encoded).long()
        
        return X_tensor


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class TokenizerSentencePiece:
    def __init__(self, vocab_size=8000, model_type="bpe", max_length=100):
        self.vocab_size_ = vocab_size
        self.model_type = model_type
        self.max_length = max_length
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.sp_ = None
        self._tmp_prefix_ = None

    
    def fit(self, X, y=None):
        texts = X.iloc[:, 0].astype(str).tolist()

        # cria arquivo temporário automaticamente
        tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_prefix_ = os.path.join(tmp_dir.name, "spm")
        input_path = os.path.join(tmp_dir.name, "train.txt")

        with open(input_path, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line.strip() + "\n")

        # treina SentencePiece (gera arquivos temporários automaticamente)
        spm.SentencePieceTrainer.train(
            input=input_path,
            model_prefix=self._tmp_prefix_,
            vocab_size=self.vocab_size_,
            model_type=self.model_type,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            hard_vocab_limit=False,
            character_coverage=1.0
        )

        # carrega direto para memória
        self.sp_ = spm.SentencePieceProcessor(model_file=self._tmp_prefix_ + ".model")
        # guarda referência para evitar garbage collection
        self._tmp_dir_ = tmp_dir

        return self

    
    def encode(self, sentences):
        X_indices = np.full((len(sentences), self.max_length + 2), fill_value=self.pad_id, dtype=np.int64)

        for i, sentence in enumerate(sentences):
            tokens = self.sp_.encode(sentence, out_type=int)
            tokens = [self.bos_id] + tokens[:self.max_length] + [self.eos_id]
            X_indices[i, :len(tokens)] = tokens

        return X_indices

    
    def decode(self, X):
        sentences = []

        for row in X.tolist():
            tokens = []

            for idx in row:
                if idx == self.eos_id:
                    break
                if idx in (self.pad_id, self.bos_id):
                    continue
                tokens.append(idx)

            sentences.append(self.sp_.decode(tokens))

        return sentences

    
    def transform(self, X):
        texts = X.iloc[:, 0].astype(str).tolist()

        lengths = [len(self.sp_.encode(t)) for t in texts]

        print("SentencePiece")
        print(np.mean(lengths))
        print(np.percentile(lengths,[50,90,95,99]))

        X_encoded = self.encode(texts)

        return torch.from_numpy(X_encoded).long()


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)