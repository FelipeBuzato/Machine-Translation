class DataSplitter:
    def __init__(self, src_language, target_language):
        self.src_language = src_language
        self.target_language = target_language

    
    def split(self, X):
        return X[[self.src_language]].copy(), X[[self.target_language]].copy()
        