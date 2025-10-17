import yaml, types

def load_yaml(path: str) -> dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def ns(**kw):
    return types.SimpleNamespace(**kw)
