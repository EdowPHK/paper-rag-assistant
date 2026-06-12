from typing import List, Dict

from schemas import RetrievalResult

def rrf_fuse_results(
    ranked_lists: List[List[RetrievalResult]],
    top_k: int = 5,
    k: int = 60,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    fused: Dict[str, RetrievalResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["retrieval_source"] = "hybrid"
            fused[chunk_id]["rrf_score"] = float(fused[chunk_id].get("rrf_score", 0.0)) + 1 / (k + rank)
            if result.get("retrieval_source") == "vector":
                fused[chunk_id]["vector_score"] = float(result.get("score", 0.0))
            if result.get("retrieval_source") == "bm25":
                fused[chunk_id]["bm25_score"] = float(result.get("score", 0.0))

    results = list(fused.values())
    for result in results:
        result["score"] = float(result.get("rrf_score", 0.0))
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]