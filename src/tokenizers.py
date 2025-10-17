class EnglishTokenizer:
    def __call__(self, text: str):
        import re
        return re.findall(r"\b\w\w+\b", text.lower())

class MeCabTokenizer:
    def __init__(self, dict_type='unidic'):
        try:
            import MeCab  # noqa
        except Exception:
            pass
    def __call__(self, text: str):
        try:
            import fugashi  # optional
            tagger = fugashi.Tagger()
            toks = []
            for w in tagger(text):
                lemma = getattr(w.feature, 'lemma', None)
                toks.append(lemma or w.surface)
            return toks
        except Exception:
            return text.split()
