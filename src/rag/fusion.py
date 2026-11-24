def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merge several ranked id lists without comparing their raw scores."""
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
