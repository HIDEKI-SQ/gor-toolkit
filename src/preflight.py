import os, re, json

class PreflightError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

def preflight(config: dict) -> None:
    # E100: preregistration
    if not ("preregister" in config or os.path.exists("manifest.yaml")):
        raise PreflightError("E100", "前登録情報が見つかりません。")

    # E110: eval_mode == union_k
    eval_mode = (config.get("preregister") or {}).get("eval_mode")
    if eval_mode != "union_k":
        raise PreflightError("E110", f"評価モード '{eval_mode}' は許可されていません。")

    # E120: reversible-only postprocess
    allowed = {"normalize_case", "strip_punct", "lexicon_isomorph", "date_normalize"}
    post = set((config.get("preregister") or {}).get("phi_preserve", []))
    invalid = list(post - allowed)
    if invalid:
        raise PreflightError("E120", f"不正な後処理: {invalid}")

    # E130: complete log
    log = dict(config.get("log", {}))
    if "prompt" not in log:  # auto-complement
        log["prompt"] = "N/A"
    required = ["seed", "model_id", "hash"]
    missing = [k for k in required if k not in log]
    if missing:
        raise PreflightError("E130", f"完全ログに不足: {missing}")

    # E140: PII minimal scan
    config_str = json.dumps(config)
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        r'\b\d{2,4}-\d{2,4}-\d{4}\b'
    ]
    if any(re.search(p, config_str) for p in patterns):
        raise PreflightError("E140", "個人情報（PII）が検出されました。")
