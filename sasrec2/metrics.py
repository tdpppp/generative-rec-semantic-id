import math


def ranking_metrics(ranks, total, prefix=""):
    hits = len(ranks)
    values = {
        "users": total,
        "hits": hits,
        "hr": hits / total if total else 0.0,
        "recall": hits / total if total else 0.0,
        "ndcg": sum(1 / math.log2(rank + 1) for rank in ranks) / total if total else 0.0,
        "mrr": sum(1 / rank for rank in ranks) / total if total else 0.0,
    }
    return {f"{prefix}{key}" if prefix else key: value for key, value in values.items()}
