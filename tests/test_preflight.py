from src.preflight import preflight, PreflightError

def test_mode_must_be_union_k():
    cfg = {
        "preregister": {"eval_mode": "intersect_k"},
        "log": {"seed": 42, "model_id": "tfidf-v1", "hash": "sha256:" + "a"*64}
    }
    try:
        preflight(cfg)
        assert False, "Should raise"
    except PreflightError as e:
        assert e.code == "E110"
