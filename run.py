#!/usr/bin/env python
import sys, json
from src.utils import load_yaml, ns
from src.preflight import preflight
from src.union_k import UnionKEncoder
from src.q1_rate_distortion import compute_q1
from src.q2_skeleton_anchor import compute_q2
from src.q3_change_visibility import compute_q3
from src.q4_anchor_guidance import compute_q4

def main(yaml_path):
    data = load_yaml(yaml_path)
    print(f"[INFO] Loading input: {yaml_path}")
    print("[INFO] Starting Preflight checks...")
    preflight(data)
    print("[INFO] Preflight checks passed")

    lang = data["document"]["lang"]
    enc = UnionKEncoder(lang=lang, seed=data["log"]["seed"])

    corpus = []
    for n in data["skeleton"]["nodes"]:
        corpus.append((n["gist"] + " " + n["detail"]).strip())
    enc.fit(corpus)
    print("[INFO] Union-K encoding completed")

    gist_all = " ".join(n["gist"] for n in data["skeleton"]["nodes"])
    detail_all = " ".join(n["detail"] for n in data["skeleton"]["nodes"])
    B = ns(gist=gist_all, detail=detail_all)
    T = ns(gist=gist_all, detail=detail_all)

    q1 = compute_q1(B, T, enc)
    print(f"[INFO] Q1 (Rate-Distortion): rho={q1['rho']:.3f}, L_U={q1['L_U']:.3f}, D={q1['D']:.3f}")

    q2 = compute_q2(data["skeleton"], data["anchors"])
    print(f"[INFO] Q2 (Skeleton-Anchor): A_sa={q2['A_sa']:.3f}")

    q3 = compute_q3(data["anchors"]["transitions"], data["skeleton"], enc)
    print(f"[INFO] Q3 (Change Visibility): V_ch={q3['V_ch']:.3f}")

    q4 = compute_q4(data["skeleton"], data["anchors"])
    print(f"[INFO] Q4 (Anchor Guidance): G_a={q4['G_a']:.2f}")

    out = {
        "doc_id": data["document"]["id"],
        "version": "gor-toolkit-0.1.1",
        "metrics": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4},
        "eval": {"mode": data["preregister"]["eval_mode"], "encoder": "tfidf-v1", "lang": lang},
        "log": {k: data["log"][k] for k in ["seed","model_id","hash"] if k in data["log"]},
    }
    out_path = yaml_path.replace(".yaml", "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Results saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py examples/mwe_a_2025-109.yaml")
        sys.exit(1)
    main(sys.argv[1])
