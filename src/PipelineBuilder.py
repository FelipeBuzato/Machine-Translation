from Trainer import Trainer
from DataCleaner import DataCleaner
from Tokenizer import TokenizerFullWords, TokenizerSentencePiece
from ModelCollection import ModelCollection
from sklearn.base import BaseEstimator
import sacrebleu
import torch
import time

class Pipeline(BaseEstimator):
    def __init__(self, lowercase=None, tokenizer_method="sentencepiece", min_freq=None, max_length=None, vocab_size=None,
                 model_name="LSTM", embedding_dim=None, hidden_dim=None, num_layers=None, dropout=None, teacher_forcing=None, max_length_decoded=None,
                 optimizer=None, lr=None, epochs=None, batch_size=None, criterion=None, random_state=42, device=None):
        
        # data cleaner hyper-parameters
        self.lowercase = lowercase

        # tokenizer hyper-parameters
        self.tokenizer_method = tokenizer_method
        # full words
        self.min_freq = min_freq
        # sentence piece
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # model hyper-parameters
        self.model_name = model_name
        # LSTM
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.teacher_forcing = teacher_forcing
        self.max_length_decoded = max_length_decoded
        
        # trainer hyper-parameters
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion

        self.random_state = random_state
        self.device = device

        self.data_cleaner_src_ = None
        self.data_cleaner_target_ = None
        self.tokenizer_src_ = None
        self.tokenizer_target_ = None
        self.model_collection_ = None
        self.model_ = None
        self.trainer_ = None
        

    def fit(self, X, y):
        start = time.time()
        # clean data
        self.data_cleaner_src_ = DataCleaner(lowercase=self.lowercase)
        self.data_cleaner_target_ = DataCleaner(lowercase=self.lowercase)
        X = self.data_cleaner_src_.fit_transform(X)
        y = self.data_cleaner_target_.fit_transform(y)
        print("Time cleaning: ", time.time()-start)

        # tokenize data
        start = time.time()
        self.tokenizer_src_ = self.get_tokenizer()
        self.tokenizer_target_ = self.get_tokenizer()
        X = self.tokenizer_src_.fit_transform(X)
        y = self.tokenizer_target_.fit_transform(y)
        print("Time tokenizing: ", time.time()-start)

        # get model
        model_params = self.get_model_params()
        self.model_collection_ = ModelCollection()
        self.model_ = self.model_collection_.get(self.model_name, model_params)

        # train model
        trainer_params = self.get_trainer_params()
        self.trainer_ = Trainer(**trainer_params)
        self.trainer_.fit(self.model_, X, y)

        return self


    def predict(self, X):
        start = time.time()
        X = self.data_cleaner_src_.transform(X)
        X = self.tokenizer_src_.transform(X)
        print("time cleaning and encoding test: ", time.time()-start)

        start = time.time()
        preds = self.trainer_.predict(self.model_, X)
        print("time predicting test: ", time.time()-start)

        start = time.time()
        preds = self.tokenizer_target_.decode(preds)
        print("Time decodig test: ", time.time()-start)

        return preds


    def score(self, X, y):
        predictions = self.predict(X)
        return self.accuracy(predictions, y)
    

    def accuracy(self, y_pred, y_true):
        y_true = self.data_cleaner_target_.transform(y_true)
        true_sentences = y_true.iloc[:, 0].to_list()
        bleu = sacrebleu.corpus_bleu(y_pred, [true_sentences], force=True)
        return bleu.score

    
    def _get_subset_params(self, keys):
        # get all params that aren't None
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
    

    def get_model_params(self):
        if(self.model_name in ["LSTM"]):
            keys = ["embedding_dim", "hidden_dim", "num_layers", "dropout", "teacher_forcing", "max_length_decoded"]
            params = self._get_subset_params(keys)
            params["vocab_size_src"] = self.tokenizer_src_.vocab_size_
            params["vocab_size_target"] = self.tokenizer_target_.vocab_size_
            return params
        
        else:
            raise ValueError(f"Model name {self.model_name} not found.")
    

    def get_trainer_params(self):
        return self._get_subset_params(["optimizer", "lr", "epochs", "batch_size", "criterion", "random_state", "device"])


    def get_tokenizer(self):
        if(self.tokenizer_method == "full words"):
            keys = ["min_freq", "max_length"]
            params = self._get_subset_params(keys)
            return TokenizerFullWords(**params)
        
        elif(self.tokenizer_method == "sentencepiece"):
            keys = ["max_length", "vocab_size"]
            params = self._get_subset_params(keys)
            return TokenizerSentencePiece(**params)
        
        else: raise ValueError(f"Tokenizer method {self.tokenizer_method} not found.")
    

    @property
    def named_steps(self):
        return {
            "data_cleaner_src": self.data_cleaner_src_,
            "data_cleaner_target": self.data_cleaner_target_,
            "tokenizer_src": self.tokenizer_src_,
            "tokenizer_target": self.tokenizer_target_,
            "model": self.model_,
        }
    

    def reset(self):
        self.model_collection_ = None
        self.data_cleaner_src_ = None
        self.data_cleaner_target_ = None
        self.tokenizer_src_ = None
        self.tokenizer_target_ = None
        self.model_ = None
        self.trainer_ = None

    
    def save(self, path):
        checkpoint = {
            "model_state_dict": self.model_.state_dict(),
            "params": self.get_params(),
            "tokenizer_src": self.tokenizer_src_,
            "tokenizer_target": self.tokenizer_target_,
            "data_cleaner_src": self.data_cleaner_src_,
            "data_cleaner_target": self.data_cleaner_target_,
            "trainer_params": self.get_trainer_params(),
        }
        torch.save(checkpoint, path)

    
    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        pipeline = cls(**checkpoint["params"])
        pipeline.data_cleaner_src_ = checkpoint["data_cleaner_src"]
        pipeline.data_cleaner_target_ = checkpoint["data_cleaner_target"]
        pipeline.tokenizer_src_ = checkpoint["tokenizer_src"]
        pipeline.tokenizer_target_ = checkpoint["tokenizer_target"]
        pipeline.trainer_ = Trainer(**checkpoint["trainer_params"])

        model_params = pipeline.get_model_params()
        pipeline.model_collection_ = ModelCollection()
        pipeline.model_ = pipeline.model_collection_.get(pipeline.model_name, model_params)
        pipeline.model_.load_state_dict(checkpoint["model_state_dict"])
        pipeline.model_.to(pipeline.trainer_.device)

        return pipeline