import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .tokenizers import MeCabTokenizer, EnglishTokenizer

class UnionKEncoder:
    def __init__(self, lang='ja', seed=42):
        self.lang = lang
        self.seed = seed
        self.vectorizer = self._init_vectorizer()
        self.vocabulary_ = None

    def _init_vectorizer(self):
        if self.lang == 'ja':
            tokenizer = MeCabTokenizer()
        elif self.lang == 'en':
            tokenizer = EnglishTokenizer()
        else:
            raise ValueError(f"Unsupported language: {self.lang}")
        return TfidfVectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            dtype=np.float32,
        )

    def fit(self, corpus):
        self.vectorizer.fit(corpus)
        self.vocabulary_ = self.vectorizer.vocabulary_
        return self

    def encode(self, gist: str, detail: str):
        if self.vocabulary_ is None:
            raise RuntimeError('Encoder not fitted.')
        K_gist = self.vectorizer.transform([gist]).toarray()[0]
        K_detail = self.vectorizer.transform([detail]).toarray()[0]
        return K_gist, K_detail

    def encode_union(self, gist: str, detail: str):
        Kg, Kd = self.encode(gist, detail)
        return np.maximum(Kg, Kd)
