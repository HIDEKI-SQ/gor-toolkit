import numpy as np

def _normalize(vec, eps=1e-10):
    v = np.maximum(vec, 0) + eps
    return v / v.sum()

def _kl(P, Q, eps=1e-10):
    P, Q = _normalize(P, eps), _normalize(Q, eps)
    return float(np.sum(P * np.log(P / Q)))

def compute_q3(transitions: list, skeleton: dict, encoder):
    node_map = {n["id"]: n for n in skeleton["nodes"]}
    vals = []
    for t in transitions:
        nid = t.get("at")
        if not nid or nid not in node_map: continue
        n = node_map[nid]
        Kg = encoder.vectorizer.transform([n["gist"]]).toarray()[0]
        Ku = encoder.encode_union(n["gist"], n["detail"])
        vals.append(_kl(Kg, Ku))
    V_ch = float(np.mean(vals)) if vals else 0.0
    return {"V_ch": V_ch, "deltas": vals, "num_transitions": len(vals)}
