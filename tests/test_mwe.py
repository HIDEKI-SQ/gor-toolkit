from types import SimpleNamespace
from src.union_k import UnionKEncoder
from src.q1_rate_distortion import compute_q1

def test_q1_smoke():
    enc = UnionKEncoder(lang='ja', seed=42).fit(["報告の幾何学 は 測定可能", "理論 を 検証"])
    B = SimpleNamespace(gist="報告の幾何学", detail="は 測定可能")
    T = SimpleNamespace(gist="理論", detail="を 検証")
    res = compute_q1(B, T, enc)
    assert 0.0 <= res["L_U"] <= 1.0
