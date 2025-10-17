import numpy as np, networkx as nx
from .q2_skeleton_anchor import _build_dag, _key_nodes

def compute_q4(skeleton: dict, anchors: dict):
    G = _build_dag(skeleton)
    V_star = _key_nodes(G)
    anchor_nodes = [t["at"] for t in anchors.get("transitions", []) if "at" in t]

    def mean_path(srcs):
        L = []
        for s in srcs:
            for v in V_star:
                if s in G and v in G and nx.has_path(G, s, v):
                    L.append(nx.shortest_path_length(G, s, v))
        return float(np.mean(L)) if L else 0.0

    baseline = mean_path(G.nodes)
    with_anchors = mean_path(anchor_nodes)
    return {"G_a": float(baseline - with_anchors),
            "baseline": float(baseline),
            "with_anchors": float(with_anchors)}
