from typing import List

from config import get_config
from schemas import AnswerResult, AnswerSource
from retrieval import retrieve_with_rerank
from generation import build_context, build_prompt, call_llm

def answering(
    query: str,
    candidate_k: int | None = None,
    embed_model: str | None = None,
    rerank_model: str | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
    max_context_chars: int | None = None,
) -> AnswerResult:
    cfg = get_config()
    candidate_k = candidate_k if candidate_k is not None else int(cfg.get("rerank_candidate_k", 20))
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    collection_name = str(collection_name or cfg.get("collection_name"))
    embed_model = str(embed_model or cfg.get("embed_model"))
    rerank_model = str(rerank_model or cfg.get("rerank_model"))
    max_context_chars = max_context_chars if max_context_chars is not None else int(cfg.get("max_context_chars", 6000))

    rerank_results = retrieve_with_rerank(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        collection_name=collection_name,
        model=embed_model,
        rerank_model=rerank_model,
    )
    context = build_context(rerank_results, max_chars=max_context_chars)
    prompt = build_prompt(context, query)
    answer_text = call_llm(prompt)

    sources: List[AnswerSource] = [
        {
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
            "heading": item["heading"],
            "score": item["score"],
            "retrieval_source": item["retrieval_source"],
        }
        for item in rerank_results
    ]
    return {
        "query": query,
        "answer": answer_text,
        "sources": sources,
    }
