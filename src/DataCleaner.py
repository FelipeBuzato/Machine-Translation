import re
from bs4 import BeautifulSoup
import html

class DataCleaner:
    def __init__(self, lowercase=False):
        self.lowercase = lowercase
        self.cols_to_drop = []


    def _clean_text(self, text):

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text(" ")

        # Replace non-breaking spaces with regular spaces
        text = text.replace("\xa0", " ")

        # Replace control characters with spaces
        text = re.sub(r"[\n\r\t\f\v]+", " ", text)

        # Collapse multiple spaces
        text = re.sub(r" +", " ", text)

        # put sentence in lowercase
        if self.lowercase: text = text.lower()

        # this cleaning keeps: punctuation, apostrophes, hyphens and accents

        return text.strip()


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        X = X.copy()
                
        X = X.drop(columns=self.cols_to_drop)

        # clean sentences
        for col in X.columns.tolist():
            X[col] = X[col].apply(self._clean_text)

        return X 


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)