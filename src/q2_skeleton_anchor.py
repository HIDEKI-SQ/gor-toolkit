"""
Q2: Skeleton–Anchor Agreement

二層メタ文書（skeleton/anchors）から、骨格とアンカーの整合度を計算する。

指標 A_sa は、
  A_sa = alpha * coverage - beta * mean_path_norm
で定義する。

- coverage: アンカー節点から到達可能な「要点節点（V*）」の被覆率
- mean_path_norm: アンカー→要点への平均最短距離をグラフの“直径”で正規化

注意:
  - 文書骨格は一般に DAG（非巡回有向グラフ）であり「強連結」ではない。
    そのため networkx の直径計算を素直に呼ぶとエラーになり得る。
  - 本実装では、弱連結判定や全ペア最短路から有限距離の最大値をとるなど、
    DAG に対しても落ちない“安全な直径”を定義して用いる。
"""

from __future__ import annotations
import networkx as nx
import numpy as np
from typing import Dict, Set


# ---------------------------------------------------------------------
# ユーティリティ: 骨格DAGの構築と要点節点の抽出
# ---------------------------------------------------------------------

def _build_dag(skeleton: Dict) -> nx.DiGraph:
    """skeleton dict から有向グラフを構築する。"""
    G = nx.DiGraph()
    for n in skeleton.get("nodes", []):
        G.add_node(n["id"], **n)
    for e in skeleton.get("edges", []):
        G.add_edge(e["from"], e["to"], dep=e.get("dep", "reference"))
    return G


def _key_nodes(G: nx.DiGraph) -> Set[str]:
    """要点節点 V* を抽出する。

    定義（実装上の経験則）:
      - type が 'claim' or 'result'
      - 出次数が 2 以上（分岐）
      - 入次数が 0（起点）
    """
    V_star: Set[str] = set()
    for nid in G.nodes:
        t = G.nodes[nid].get("type")
        if t in {"claim", "result"}:
            V_star.add(nid)
            continue
        if G.out_degree(nid) >= 2:
            V_star.add(nid)
            continue
        if G.in_degree(nid) == 0:
            V_star.add(nid)
    return V_star


# ---------------------------------------------------------------------
# メイン: Q2（二層整合度）
# ---------------------------------------------------------------------

def compute_q2(
    skeleton: Dict,
    anchors: Dict,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Dict:
    """Q2（二層整合度）を計算して dict で返す。"""

    # --- グラフ構築 ---
    G = _build_dag(skeleton)
    V_star = _key_nodes(G)  # 要点節点集合

    # --- アンカー節点 ---
    anchor_nodes = [t.get("at") for t in anchors.get("transitions", []) if t.get("at")]
    anchor_nodes = [a for a in anchor_nodes if a in G]  # 存在するものだけ

    # --- 被覆率 coverage ---
    covered: Set[str] = set()
    for a in anchor_nodes:
        covered |= (nx.descendants(G, a) | {a})
    coverage = len(covered & V_star) / max(1, len(V_star))

    # --- アンカー→要点への最短距離 ---
    path_lengths = []
    for a in anchor_nodes:
        for v in V_star:
            if nx.has_path(G, a, v):
                path_lengths.append(nx.shortest_path_length(G, a, v))
    mean_path = float(np.mean(path_lengths)) if path_lengths else 0.0

    # --- “安全な直径”の定義 ---
    # DAG（強連結でない）でも落ちないように:
    #  1) 弱連結なら無向化して diameter を計算
    #  2) それ以外は全ペア最短距離の有限値最大を採用
    try:
        if len(G) <= 1:
            D = 1
        elif nx.is_weakly_connected(G):
            D = nx.diameter(G.to_undirected())
        else:
            all_pairs = dict(nx.all_pairs_shortest_path_length(G))
            finite = [d for dists in all_pairs.values() for d in dists.values()]
            D = max(finite) if finite else 1
    except Exception:
        D = max(1, len(G.nodes))

    mean_path_norm = mean_path / max(1, D)

    # --- 整合度 A_sa ---
    A_sa = alpha * coverage - beta * mean_path_norm

    return {
        "A_sa": float(A_sa),
        "coverage": float(coverage),
        "mean_path": float(mean_path_norm),
        "details": {
            "num_key_nodes": int(len(V_star)),
            "num_anchors": int(len(anchor_nodes)),
            "raw_mean_path": float(mean_path),
            "diameter": float(D),
        },
    }
