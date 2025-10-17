import numpy as np, networkx as nx

def _build_dag(skeleton: dict):
    G = nx.DiGraph()
    for n in skeleton["nodes"]:
        G.add_node(n["id"], **n)
    for e in skeleton["edges"]:
        G.add_edge(e["from"], e["to"], dep=e["dep"])
    return G

def _key_nodes(G):
    V = set()
    for nid in G.nodes:
        t = G.nodes[nid].get("type")
        if t in {"claim", "result"} or G.out_degree(nid) >= 2 or G.in_degree(nid) == 0:
            V.add(nid)
    return V

def compute_q2(skeleton: dict, anchors: dict, alpha=0.7, beta=0.3):
    G = _build_dag(skeleton)
    V_star = _key_nodes(G)
    anchor_nodes = [t["at"] for t in anchors.get("transitions", []) if "at" in t]

    covered = set()
    for a in anchor_nodes:
        if a in G:
            covered |= (nx.descendants(G, a) | {a})
    coverage = len(covered & V_star) / max(1, len(V_star))

    dists = []
    for a in anchor_nodes:
        for v in V_star:
            if a in G and v in G and nx.has_path(G, a, v):
                dists.append(nx.shortest_path_length(G, a, v))
    mean_path = np.mean(dists) if dists else 1.0
    diameter = nx.diameter(G) if nx.is_weakly_connected(G) else max(1, len(G.nodes))
    mean_path_norm = mean_path / diameter

    A_sa = alpha * coverage - beta * mean_path_norm
    return {"A_sa": float(A_sa), "coverage": float(coverage), "mean_path": float(mean_path_norm)}
