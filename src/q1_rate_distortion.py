import numpy as np

def cosine_similarity(v1, v2):
    dot = float(np.dot(v1, v2))
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0: return 0.0
    return max(0.0, min(1.0, dot/(n1*n2)))

def _estimate_distortion_at_rho(rho_target, rho_actual, D_actual):
    D_min, D_max, EPS = 0.05, 1.0, 1e-12
    ratio = (D_actual - D_min) / max(EPS, (D_max - D_min))
    if rho_actual <= EPS or ratio <= EPS:
        beta = 1.0
    else:
        beta = -np.log(ratio) / rho_actual
    D_est = D_min + (D_max - D_min) * np.exp(-beta * rho_target)
    return float(np.clip(D_est, 0.0, 1.0))

def compute_q1(B, T, encoder, rho_range=np.linspace(0.1, 0.9, 9)):
    analyzer = encoder.vectorizer.build_analyzer()
    len_B = len(analyzer(B.gist + " " + B.detail))
    len_T = len(analyzer(T.gist + " " + T.detail))
    rho_actual = len_B / max(1, len_T)

    KBg, KBd = encoder.encode(B.gist, B.detail)
    KBgT, KBdT = encoder.encode(T.gist, T.detail)
    KB_union = np.maximum(KBg, KBd)
    KT_union = np.maximum(KBgT, KBdT)

    L_U = cosine_similarity(KB_union, KT_union)
    D = 1.0 - L_U

    curve = [(float(r), _estimate_distortion_at_rho(r, rho_actual, D)) for r in rho_range]
    return {"rho": float(rho_actual), "L_U": float(L_U), "D": float(D), "curve": curve}
