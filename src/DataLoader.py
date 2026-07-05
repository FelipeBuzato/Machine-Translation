import pandas as pd

class DataLoader:
    def __init__(self, path_train, path_eval, path_test):
        self.path_train = path_train
        self.path_eval = path_eval
        self.path_test = path_test


    def load(self):
        train_data = pd.read_csv(self.path_train)
        eval_data = pd.read_csv(self.path_eval)
        test_data = pd.read_csv(self.path_test)
        return train_data, eval_data, test_data